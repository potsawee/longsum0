import os
import sys
sys.path.insert(0, os.getcwd()+'/data/') # to import modules in data
sys.path.insert(0, os.getcwd()+'/models/') # to import modules in models

import pickle
import random
import argparse
import torch
import numpy as np
from nltk import tokenize
from transformers import BartTokenizer, BertTokenizer
from podcast_processor import PodcastEpisode
from arxiv_processor import ResearchArticle
from hiermodel import EncoderDecoder, EXTLabeller

def inference_hiermodel(args):
    start_id   = args['start_id']
    end_id     = args['end_id']
    decode_dir = args['decode_dir']
    inference_mode = args['inference_mode']

    # uses GPU in training or not
    if torch.cuda.is_available() and args['use_gpu']: torch_device = 'cuda'
    else: torch_device = 'cpu'
    use_gpu = args['use_gpu']

    # ----- Hierarchical Model Configurations ----- #
    # TODO: replace this part to be not hard coded
    args['num_utterances'] = args['max_num_sent']
    args['num_words']      = args['max_word_in_sent']

    args['vocab_size']     = 30522 # BERT tokenizer
    args['embedding_dim']   = 256   # word embeeding dimension
    args['rnn_hidden_size'] = 512 # RNN hidden size
    args['dropout']        = 0.0
    args['num_layers_enc'] = 2    # in total it's num_layers_enc*2 (word/utt)
    args['num_layers_dec'] = 1
    # --------------------------------------------- #

    # Load the model
    trained_model_path = args['load']
    if use_gpu:
        state = torch.load(trained_model_path)
    else:
        state = torch.load(trained_model_path, map_location=torch.device('cpu'))
    model = EncoderDecoder(args, device=torch_device)
    model_state_dict = state['model']
    model.load_state_dict(model_state_dict)

    ext_labeller = EXTLabeller(rnn_hidden_size=args['rnn_hidden_size'], dropout=args['dropout'], device=torch_device)
    if inference_mode in ['ext', 'mcs']:
        ext_labeller_state_dict = state['ext_labeller']
        ext_labeller.load_state_dict(ext_labeller_state_dict)
    model.eval()
    ext_labeller.eval()
    print('model loaded!')

    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # data
    # datapath = "podcast_sum0/lib/test_data/podcast_testset.bin"
    # datapath = "arxiv_sum0/lib/data/arxiv_test.pk.bin"
    # datapath = "arxiv_sum0/lib/pubmed_data/pubmed_test.pk.bin"
    datapath = args['datapath']
    print("datapath =", datapath)
    with open(datapath, 'rb') as f:
        data = pickle.load(f, encoding="bytes")
    print("len(data) = {}".format(len(data)))

    ids = [x for x in range(start_id, end_id)]
    if args['random_order']: random.shuffle(ids)

    # inference parameters
    beam_width    = args['beam_width']
    time_step     = args['time_step']
    penalty_ug    = args['penalty_ug']
    alpha         = args['alpha']
    length_offset = args['length_offset']

    for id in ids:
        # check if the file exist or not
        out_path = "{}/{}_decoded.pk.bin".format(decode_dir, id)
        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(id))
            continue

        if args['dataset'] == 'podcast':
            input_text = data[id].transcription
            sentences = tokenize.sent_tokenize(input_text)
        elif args['dataset'] == 'arxiv' or args['dataset'] == 'pubmed':
            input_text = " ".join(data[id].article_text)
            sentences  = data[id].article_text
        else:
            raise ValueError("Dataset not exist: only |podcast|arxiv|pubmed|")

        num_sent = len(sentences)

        try: l1 = len(bart_tokenizer.encode(input_text, max_length=500000))
        except IndexError: l1 = 0

        # the length is within the limit --> no selection needed
        if l1 < args['max_abssum_len']:
            filtered_sentences = sentences
        # perform MCS
        else:
            keep_idx = []
            input, u_len, w_len = get_enc_input(bert_tokenizer, [sentences], args['max_num_sent'],
                                                args['max_word_in_sent'], use_gpu=use_gpu)
            # ------ MODULE1: Extractive Sum ------ #
            # Forward pass
            with torch.no_grad():
                encoder_output_dict = model.encoder(input, u_len, w_len)
                enc_u_output = encoder_output_dict['u_output']
                ext_output = ext_labeller(enc_u_output).squeeze(-1)
            ext_output = ext_output[0].cpu().numpy()
            # -------------- END MODULE1 --------------- #
            # ------ MODULE2: Sentence-Level Attn ------ #
            batch = {"input": input, "u_len": u_len, "w_len": w_len}
            attention = get_utt_attn_without_ref(model, batch, beam_width=beam_width, time_step=time_step,
                        penalty_ug=penalty_ug, alpha=alpha, length_offset=length_offset, torch_device=torch_device)
            if len(sentences) != attention.shape[0]:
                if len(sentences) > args['max_num_sent']:
                    sentences = sentences[:args['max_num_sent']]
                else:
                    raise ValueError("shape error #1")
            # -------------- END MODULE2 --------------- #
            N1 = len(attention)
            N2 = len(ext_output)
            if N2 > N1: ext_output = ext_output[:N1]
            ext_score = compute_ranking_score(ext_output)
            attn_score = compute_ranking_score(attention)

            if inference_mode == 'mcs':
                # taking geometric mean --- (simple mean works too!)
                total_score = np.sqrt(attn_score * ext_score)
            elif inference_mode == 'ext':  total_score = ext_score
            elif inference_mode == 'attn': total_score = attn_score
            else: raise Exception("inference mode error!")

            rank = np.argsort(total_score)[::-1]
            keep_idx = []
            total_length = 0
            for sent_i in rank:
                if total_length < args['max_abssum_len']:
                    sent = sentences[sent_i]
                    length = len(bart_tokenizer.encode(sent, max_length=50000)[1:-1]) # ignore <s> and </s>
                    total_length += length
                    keep_idx.append(sent_i)
                else:
                    break
            keep_idx = sorted(keep_idx)
            filtered_sentences = [sentences[j] for j in keep_idx]
        with open(out_path, "wb") as f:
            pickle.dump(filtered_sentences, f)
        print("write:", out_path)


def get_utt_attn_without_ref(model, enc_batch, beam_width=4, time_step=240,
                            penalty_ug=0.0, alpha=1.25, length_offset=5, torch_device='cpu'):
    decode_dict = {
        'k': beam_width,
        'time_step': time_step,
        'vocab_size': 30522,
        'device': torch_device,
        'start_token_id': 101, 'stop_token_id': 103,
        'alpha': alpha,
        'length_offset': length_offset,
        'penalty_ug': penalty_ug,
        'keypadmask_dtype': torch.bool,
        'memory_utt': False,
        'batch_size': 1
    }
    # batch_size should be 1
    with torch.no_grad():
        summary_ids, attn_scores, u_attn_scores = model.decode_beamsearch(
                enc_batch["input"], enc_batch["u_len"], enc_batch["w_len"], decode_dict)

    N = enc_batch["u_len"][0].item()
    attention = u_attn_scores[:,:N].sum(dim=0) / u_attn_scores[:,:N].sum()
    attention = attention.cpu().numpy()
    return attention

def get_enc_input(bert_tokenizer, list_sentences,
        max_num_sent, max_word_in_sent, use_gpu=False):

    batch_size = len(list_sentences)
    input = np.zeros((batch_size, max_num_sent, max_word_in_sent), dtype=np.int64)
    u_len = np.zeros((batch_size), dtype=np.int64)
    w_len = np.zeros((batch_size, max_num_sent), dtype=np.int64)

    for i, sentences in enumerate(list_sentences):
        num_sentences = len(sentences)
        if num_sentences > max_num_sent:
            num_sentences = max_num_sent
            sentences = sentences[:max_num_sent]
        u_len[i] = num_sentences

        for j, sent in enumerate(sentences):
            token_ids = bert_tokenizer.encode(sent, max_length=500000)[1:-1] # remove [CLS], [SEP]
            utt_len = len(token_ids)
            if utt_len > max_word_in_sent:
                utt_len = max_word_in_sent
                token_ids = token_ids[:max_word_in_sent]
            input[i,j,:utt_len] = token_ids
            w_len[i,j] = utt_len
    input = torch.from_numpy(input)
    u_len = torch.from_numpy(u_len)
    w_len = torch.from_numpy(w_len)

    if use_gpu:
        input = input.cuda()
        u_len = u_len.cuda()
        w_len = w_len.cuda()

    return input, u_len, w_len

def compute_ranking_score(score):
    """
        the item with lowest rank gets 0.0, the item with highest rank gets 1.0,
        and everthing else gets the value in between 0.0 and 1.0
    """
    rank_ascending = np.argsort(score)
    N = len(score)
    if N == 1: return np.array([1.0])
    ranking_score = [None for _ in range(N)]
    for i, idx in enumerate(rank_ascending):
        ranking_score[idx] = i/(N-1)
    return np.array(ranking_score)

def get_decode_arguments(parser):
    '''Arguments for decoding'''

    parser.register("type", "bool", lambda v: v.lower() == "true")

    # file paths
    parser.add_argument('--load',       type=str, required=True)  # path to load model
    parser.add_argument('--decode_dir', type=str, required=True)
    parser.add_argument('--dataset',    type=str, required=True)
    parser.add_argument('--datapath',   type=str, required=True)
    parser.add_argument('--inference_mode', type=str, default='attn') # ext | attn | mcs

    parser.add_argument('--max_abssum_len',   type=int, default=4096)
    parser.add_argument('--max_num_sent',     type=int, default=1000)
    parser.add_argument('--max_word_in_sent', type=int, default=120)

    parser.add_argument('--beam_width',    type=int,   default=4)
    parser.add_argument('--time_step',     type=int,   default=144)
    parser.add_argument('--penalty_ug',    type=float, default=0.0)
    parser.add_argument('--alpha',         type=float, default=1.25)
    parser.add_argument('--length_offset', type=int,   default=5)

    parser.add_argument('--start_id',   type=int, required=True)
    parser.add_argument('--end_id',     type=int, required=True)
    parser.add_argument('--random_order', type="bool", nargs="?", const=True, default=False)
    parser.add_argument('--use_gpu',    type="bool", nargs="?", const=True, default=False)

    return parser

if __name__ == "__main__":
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = get_decode_arguments(parser)
    args = vars(parser.parse_args())
    inference_hiermodel(args)
