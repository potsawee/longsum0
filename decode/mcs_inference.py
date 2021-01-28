import os
import sys
import pickle
import random
import numpy as np
from tqdm import tqdm
from nltk import tokenize
from transformers import BartTokenizer

REPO = '../podcast_sum0/'
sys.path.insert(0, REPO)
from data.processor import PodcastEpisode

DATASET       = 'testset' # testset | set_dev1_brass
MODEL_NAME    = "HIER_EncDec_DEC3_v3-step30000"

DECODE_EXT_DIR  = "../summariser2/system_output_ext/{}/{}".format(MODEL_NAME, DATASET)
DECODE_ATN_DIR  = "../summariser2/system_output_attn/{}/{}".format(MODEL_NAME, DATASET)

if DATASET == 'set_dev1_brass':
    DATA_PATH  = "../podcast_sum0/lib/data/podcast_set_dev1_brass.bin"
elif DATASET == 'testset':
    DATA_PATH  = "../podcast_sum0/lib/test_data/podcast_testset.bin"

TYPE = 'geomean' # geomean | attn | ext
MAX_BART_LEN  = 4096

def compute_ranking_score(score):
    # the item with lowest rank gets 0.0, and the highest rank gets 1.0
    rank_ascending = np.argsort(score)
    N = len(score)
    if N == 1: return np.array([1.0])
    ranking_score = [None for _ in range(N)]
    for i, idx in enumerate(rank_ascending):
        ranking_score[idx] = i/(N-1)
    return np.array(ranking_score)

def main():
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # data
    print("DATA_PATH =", DATA_PATH)
    with open(DATA_PATH, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    for id in tqdm(range(len(podcasts))):
        sentences = tokenize.sent_tokenize(podcasts[id].transcription)
        num_sent = len(sentences)

        # Extractive Sum Score
        read_path = "{}/{}_ext_label.txt".format(DECODE_EXT_DIR, id)
        with open(read_path, 'r') as f:
            prob_lines = f.readlines()
        num_prob = len(prob_lines)
        if num_prob < num_sent:
            probs = [float(x.strip()) for x in prob_lines] # truncated sentences
            probs += [0.0]* (num_sent - num_sent)
        else:
            probs = [float(x.strip()) for x in prob_lines]
        ext_score = compute_ranking_score(probs)

        # Sentence-Level Attn Score
        if TYPE != 'ext':
            read_path = "{}/{}_attn.txt".format(DECODE_ATN_DIR, id)
            with open(read_path, 'r') as f:
                attn_lines = f.readlines()
            num_score = len(attn_lines)
            if num_score < num_sent:
                attn_score = [float(x.strip()) for x in attn_lines] # truncated sentences
                attn_score += [0.0]* (num_sent - num_sent)
            else:
                attn_score = [float(x.strip()) for x in attn_lines]
            attn_score = compute_ranking_score(attn_score)

            N1 = len(attn_score)
            N2 = len(ext_score)
            if N2 > N1: ext_score = ext_score[:N1]

            if TYPE == 'geomean':
                total_score = np.sqrt(attn_score * ext_score)
            elif TYPE == 'attn':
                total_score = attn_score

        elif TYPE == 'ext':
            total_score = ext_score
        else:
            raise ValueError("not defined")

        rank = np.argsort(total_score)[::-1]
        keep_idx = []
        total_length = 0
        for sent_i in rank:
            if total_length < MAX_BART_LEN:
                sent = sentences[sent_i]
                length = len(bart_tokenizer.encode(sent, max_length=50000)[1:-1]) # ignore <s> and </s>
                total_length += length
                keep_idx.append(sent_i)
            else:
                break
        keep_idx = sorted(keep_idx)
        filtered_sentences = [sentences[j] for j in keep_idx]
        filtered_transcription = " ".join(filtered_sentences)
        podcasts[id].transcription = filtered_transcription

    dump_path = "data/mcs_inference/{}/{}_type-{}_{}.bin".format(DATASET, MODEL_NAME, TYPE, MAX_BART_LEN)
    with open(dump_path, "wb") as f:
        pickle.dump(podcasts, f)
    print("dumped:", dump_path)

if __name__ == "__main__":
    main()
