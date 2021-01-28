import os
import sys
import pickle
import random
import torch
import numpy as np
from nltk import tokenize
from tqdm import tqdm

HIER_REPO = '../summariser1/'
PODS_REPO = '../podcast_sum0/'
sys.path.insert(0, HIER_REPO)
sys.path.insert(0, PODS_REPO)
from hier_model import Batch, HierTokenizer, HierarchicalModel
from data.processor import PodcastEpisode
from transformers import BartTokenizer

if torch.cuda.is_available():
    torch_device = 'cuda'
    use_gpu = True
else:
    torch_device = 'cpu'
    use_gpu = False

DATA_PATH    = "../podcast_sum0/lib/test_data/podcast_testset.bin"

MAX_BART_LEN   = 1040
MAX_INPUT_SENT = 1000
MAX_SENT_WORD  = 50

MODEL_NAME = "HGRUV5DIV_SPOTIFY_JUNE18_v3-step30000"
MODEL_PATH = "trained_models/{}.pt".format(MODEL_NAME)
DECODE_DIR = "system_output_attn/{}/{}".format(MODEL_NAME, 'testset')

def filtering_data(start_id, end_id):
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    with open(DATA_PATH, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    hier_tokenizer = HierTokenizer()
    hier_tokenizer.set_len(num_utterances=MAX_INPUT_SENT, num_words=MAX_SENT_WORD)
    hier_model = HierarchicalModel(MODEL_PATH, use_gpu=use_gpu)

    ids = [x for x in range(start_id, end_id)]
    random.shuffle(ids)

    # already do .eval() in initialisation stage

    for i in ids:
        # check if the file exist or not
        # DECODER_DIR = temp folder
        out_path = "{}/{}_attn.txt".format(DECODE_DIR, i)
        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(i))
            continue

        batch = hier_tokenizer.get_enc_input([podcasts[i].transcription], use_gpu=use_gpu)[0]
        attention = hier_model.get_utt_attn_without_ref(batch, beam_width=4, time_step=144, penalty_ug=0.0, alpha=1.25, length_offset=5)

        attention = attention.tolist()
        try:
            attn_score_str = ["{:.9f}".format(s) for s in attention]
        except TypeError:
            if isinstance(attention, float):
                attn_score_str = ["{:.9f}".format(attention)]
            else:
                import pdb; pdb.set_trace()
        with open(out_path, 'w') as file:
            file.write("\n".join(attn_score_str))
        print("write:", out_path)


if __name__ == "__main__":
    if(len(sys.argv) == 2):
        start_id = int(sys.argv[1])
        end_id   = start_id + 8
        if end_id > 1027: end_idx = 1027
        filtering_data(start_id, end_id)

    elif(len(sys.argv) == 3):
        start_id = int(sys.argv[1]) # from 0
        end_id   = int(sys.argv[2]) # to 1027
        filtering_data(start_id, end_id)
    else:
        print("Usage: python filtering_data.py start_id end_id")
        raise Exception("argv error")
