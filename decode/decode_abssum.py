import os
import sys
sys.path.insert(0, os.getcwd()+'/data/') # to import modules in data
sys.path.insert(0, os.getcwd()+'/models/') # to import modules in models

import pickle
import random
import argparse

import torch
from nltk import tokenize
from transformers import BartTokenizer, BartForConditionalGeneration
from podcast_processor import PodcastEpisode
from arxiv_processor import ResearchArticle
from localattn import LoBART

def decode(args):

    start_id   = args['start_id']
    end_id     = args['end_id']
    decode_dir = args['decode_dir']

    # uses GPU in training or not
    if torch.cuda.is_available() and args['use_gpu']: torch_device = 'cuda'
    else: torch_device = 'cpu'

    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    if args['selfattn'] == 'full':
        bart_model  = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    elif args['selfattn'] == 'local':
        window_width = args['window_width']
        xspan        = args['multiple_input_span']
        attention_window = [window_width] * 12 # different window size for each layer can be defined here too!
        bart_model = LoBART.from_pretrained('facebook/bart-large-cnn')
        bart_model.swap_fullattn_to_localattn(attention_window=attention_window)
        bart_model.expand_learned_embed_positions(multiple=xspan, cut=xspan*2)
    else:
        raise ValueError("selfattn: full (BART) | local (LoBART)")

    # trained models path
    trained_model_path = args['load']
    if torch_device == 'cuda':
        bart_model.cuda()
        state = torch.load(trained_model_path)
    else:
        state = torch.load(trained_model_path, map_location=torch.device('cpu'))
    model_state_dict = state['model']
    bart_model.load_state_dict(model_state_dict)

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

    bart_model.eval() # not needed but for sure!!

    # decoding hyperparameters
    num_beams = args['num_beams']
    length_penalty = args['length_penalty']
    max_length = args['max_length']
    min_length = args['min_length']
    no_repeat_ngram_size = args['no_repeat_ngram_size']
    for id in ids:
        # check if the file exist or not
        out_path = "{}/{}_decoded.txt".format(decode_dir, id)
        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(id))
            continue


        if args['dataset'] == 'podcast':
            input_text = data[id].transcription
        elif args['dataset'] == 'arxiv' or args['dataset'] == 'pubmed':
            input_text = " ".join(data[id].article_text)
        else:
            raise ValueError("Dataset not exist: only |podcast|arxiv|pubmed|")

        if args['selfattn'] == 'local':
            # local-attention requires sequence length to be a multiple of ...
            article_input_ids = bart_tokenizer.batch_encode_plus([input_text],
                return_tensors='pt', max_length=bart_model.config.max_position_embeddings,
                pad_to_max_length=True)['input_ids'].to(torch_device)
            summary_ids = bart_model.generate(article_input_ids,
                            num_beams=num_beams, length_penalty=length_penalty,
                            max_length=max_length, # set this equal to the max length in training
                            min_length=min_length,  # one sentence
                            no_repeat_ngram_size=no_repeat_ngram_size, pad_token_id=bart_model.config.pad_token_id)
        else: # Vanilla BART
            article_input_ids = bart_tokenizer.batch_encode_plus([input_text],
                return_tensors='pt', max_length=bart_model.config.max_position_embeddings)['input_ids'].to(torch_device)
            summary_ids = bart_model.generate(article_input_ids,
                            num_beams=num_beams, length_penalty=length_penalty,
                            max_length=max_length, # set this equal to the max length in training
                            min_length=min_length,  # one sentence
                            no_repeat_ngram_size=no_repeat_ngram_size)

        summary_txt = bart_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True).strip()
        if args['dataset'] == 'podcast':
            with open(out_path, 'w') as file:
                file.write(summary_txt)
        elif args['dataset'] == 'arxiv' or args['dataset'] == 'pubmed':
            with open(out_path, 'w') as file:
                # process it to be the same format as the reference!
                summary_sentences = tokenize.sent_tokenize(summary_txt)
                new_summary_sentences = [" ".join(tokenize.word_tokenize(sent)) for sent in summary_sentences]
                summary_out = "\n".join(new_summary_sentences)
                file.write(summary_out)
        else:
            raise ValueError("Dataset not exist: only |podcast|arxiv|pubmed|")
        print("write:", out_path)


def get_decode_arguments(parser):
    '''Arguments for decoding'''

    parser.register("type", "bool", lambda v: v.lower() == "true")

    # file paths
    parser.add_argument('--load',       type=str, required=True)  # path to load model
    parser.add_argument('--selfattn',   type=str, required=True)
    parser.add_argument('--multiple_input_span', type=int, default=None)
    parser.add_argument('--window_width', type=int, default=None)
    parser.add_argument('--decode_dir', type=str, required=True)
    parser.add_argument('--dataset',    type=str, required=True)
    parser.add_argument('--datapath',   type=str, required=True)
    parser.add_argument('--start_id',   type=int, required=True)
    parser.add_argument('--end_id',     type=int, required=True)
    parser.add_argument('--num_beams',  type=int, default=4)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=50)
    parser.add_argument('--no_repeat_ngram_size',   type=int, default=3)
    parser.add_argument('--length_penalty', type=float, default=2.0)
    parser.add_argument('--random_order', type="bool", nargs="?", const=True, default=False)
    parser.add_argument('--use_gpu',    type="bool", nargs="?", const=True, default=False)

    return parser

if __name__ == "__main__":
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = get_decode_arguments(parser)
    args = vars(parser.parse_args())
    decode(args)
