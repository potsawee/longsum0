import os
import sys
import pickle
import random
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer
from nltk import tokenize

from models.hierarchical_rnn_v5 import HierarchicalGRU, EXTLabeller
from create_podcast_extra import PodcastEpisodeXtra

REPO = '../podcast_sum0/'
sys.path.insert(0, REPO)
from data.processor import PodcastEpisode

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_NAME = "HIER_EncDec_DEC3_v3-step30000"
model_type = 'encdec' # enc | encdec

DATA_PATH          = "../podcast_sum0/lib/data/podcast_set_dev1_brass.bin"
TRAINED_MODEL_PATH = "../summariser2/lib/trained_models/{}.pt".format(MODEL_NAME)
DECODE_DIR         = "../summariser2/system_output_ext/{}/{}".format(MODEL_NAME, 'set_dev1_brass')

NUM_UTTERANCES = 1400
NUM_WORDS      = 100

def decode(start_id, end_id):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # ---------------------------------- Model ---------------------------------- #

    model = HierarchicalGRU(vocab_size=30522, embedding_dim=256, rnn_hidden_size=512,
                            num_layers=2, dropout=0.1, device=torch_device)

    ext_labeller = EXTLabeller(rnn_hidden_size=512, dropout=0.1, device=torch_device)

    if torch_device == 'cuda':
        model.cuda()
        ext_labeller.cuda()
        state = torch.load(TRAINED_MODEL_PATH)
    else:
        state = torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu'))

    if model_type == 'enc':
        model_state_dict = state['model']
    elif model_type == 'encdec':
        model_state_dict = state['model']
        new_model_state_dict = OrderedDict()
        for key in model_state_dict.keys():
            if 'encoder.' in key:
                new_model_state_dict[key.replace("encoder.","")] = model_state_dict[key]
            else: # deocde
                pass
        model_state_dict = new_model_state_dict

    ext_labeller_state_dict = state['ext_labeller']
    model.load_state_dict(model_state_dict)
    ext_labeller.load_state_dict(ext_labeller_state_dict)
    print('model loaded!')

    # data
    print("DATA_PATH =", DATA_PATH)
    with open(DATA_PATH, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    ids = [x for x in range(start_id, end_id)]
    random.shuffle(ids)

    model.eval() # not needed but for sure!!
    ext_labeller.eval() # not needed but for sure!!

    for id in ids:
        # check if the file exist or not
        out_path = "{}/{}_ext_label.txt".format(DECODE_DIR, id)
        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(id))
            continue

        # ENCODER
        input = np.zeros((1, NUM_UTTERANCES, NUM_WORDS), dtype=np.long)
        u_len = np.zeros((1), dtype=np.long)
        w_len = np.zeros((1, NUM_UTTERANCES), dtype=np.long)

        sentences = tokenize.sent_tokenize(podcasts[id].transcription)
        num_sentences = len(sentences)

        if num_sentences > NUM_UTTERANCES:
            num_sentences = NUM_UTTERANCES
            sentences = sentences[:NUM_UTTERANCES]
        u_len[0] = num_sentences

        for j, sent in enumerate(sentences):
            token_ids = bert_tokenizer.encode(sent.lower(), add_special_tokens=False)
            utt_len = len(token_ids)
            if utt_len > NUM_WORDS:
                utt_len = NUM_WORDS
                token_ids = token_ids[:NUM_WORDS]
            input[0,j,:utt_len] = token_ids
            w_len[0,j] = utt_len

        u_len_max = u_len.max()
        w_len_max = w_len.max()
        input = torch.from_numpy(input[:, :u_len_max, :w_len_max]).to(torch_device)
        u_len = torch.from_numpy(u_len).to(torch_device)
        w_len = torch.from_numpy(w_len[:, :u_len_max]).to(torch_device)

        # Forward pass
        with torch.no_grad():
            encoder_output_dict = model(input, u_len, w_len)
            enc_u_output = encoder_output_dict['u_output']
            ext_output = ext_labeller(enc_u_output).squeeze(-1)

        ext_output = ext_output.tolist()[0]
        try:
            ext_score_str = ["{:.6f}".format(s) for s in ext_output]
        except TypeError:
            if isinstance(ext_output, float):
                ext_score_str = ["{:.6f}".format(ext_output)]
            else:
                import pdb; pdb.set_trace()
        with open(out_path, 'w') as file:
            file.write("\n".join(ext_score_str))
        print("write:", out_path)
        """
        0.398045
        0.458923
        0.023489
        ...
        one sentence score per line
        """

def write_reference():
    REFERENCE_DATA_PATH = "lib/test_data/f1_oracle/podcast_ext1024_testset.bin"
    out_path = "../summariser2/reference/f1_oracle/testset.txt"

    with open(REFERENCE_DATA_PATH, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    with open(out_path, 'w') as file:
        file.write("id\tN\textractive_labels\n".format(len(podcasts)))
        for i in range(len(podcasts)):
            ext_label = podcasts[i].ext_label
            num_sent = podcasts[i].num_sent
            file.write("{}\t{}\t{}\n".format(i, num_sent, ",".join([str(x) for x in ext_label])))
    print("write:", out_path)

if __name__ == "__main__":
    if(len(sys.argv) == 2):
        start_id = int(sys.argv[1])
        end_id   = start_id + 50
        if end_id > 2189: end_idx = 2189
        # if end_id > 1027: end_idx = 1027
        decode(start_id, end_id)
    elif(len(sys.argv) == 3):
        start_id = int(sys.argv[1])
        end_id   = int(sys.argv[2])
        decode(start_id, end_id)
    else:
        print("Usage: python decode_transformer.py start_id end_id")
        raise Exception("argv error")
