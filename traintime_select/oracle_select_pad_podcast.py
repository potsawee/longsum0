import os
import sys
import pickle
import random
import torch
import numpy as np
from nltk import tokenize
from tqdm import tqdm

sys.path.insert(0, os.getcwd()+'/data/') # to import modules in data
from podcast_processor import PodcastEpisode
from transformers import BartTokenizer

from rouge import Rouge
rouge_pltrdy = Rouge()


PODCAST_SET    = 10
MAX_BART_LEN   = 8200
DATA_PATH      = "../podcast_sum0/lib/data/podcast_set{}.bin".format(PODCAST_SET)
OUTDIR         = "data/podcast/oracle-padrand/8200/"
ORC_TYPE       = 'padrand' # padrand | padlead

print("PODCAST_SET:", PODCAST_SET)
print("MAX_BART_LEN:", MAX_BART_LEN)

def get_rouge2recall_scores(sentences, reference):
    # rouge_pltrdy is case sensitive
    reference = reference.lower()
    scores = [None for _ in range(len(sentences))]
    count_nonzero_rouge2recall = 0
    for i, sent in enumerate(sentences):
        sent = sent.lower()
        try:
            rouge_scores = rouge_pltrdy.get_scores(sent, reference)
            rouge2recall = rouge_scores[0]['rouge-2']['r']
            scores[i] = rouge2recall
        except ValueError:
            scores[i] = 0.0
        except RecursionError:
            scores[i] = 0.5 # just assign 0.5 as this sentence is simply too long
        if scores[i] > 0.0: count_nonzero_rouge2recall += 1
    scores = np.array(scores)
    N = len(scores)


    if ORC_TYPE == 'padlead':
        biases = np.array([(N-i)*1e-12 for i in range(N)])
    elif ORC_TYPE == 'padrand':
        biases = np.random.normal(scale=1e-10,size=(N,))
    else:
        raise ValueError("this oracle method not supported")
    return scores + biases

def filtering_data(start_id, end_id):
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    with open(DATA_PATH, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))


    ids = [x for x in range(start_id, end_id)]
    random.shuffle(ids)

    for i in ids:
        # check if the file exist or not
        # DECODER_DIR = temp folder
        out_path = "{}/{}_filtered_transcription.txt".format(OUTDIR, i)

        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(i))
            continue

        l1 = len(bart_tokenizer.encode(podcasts[i].transcription, max_length=50000))

        if l1 < MAX_BART_LEN:
            filtered_transcription = podcasts[i].transcription

        else:
            sentences = tokenize.sent_tokenize(podcasts[i].transcription)
            reference = podcasts[i].description
            keep_idx = []

            selection_score = get_rouge2recall_scores(sentences, reference) # numpy array (80,)
            rank = np.argsort(selection_score)[::-1]
            keep_idx = []
            total_length = 0
            for sent_i in rank:
                if total_length < MAX_BART_LEN:
                    sent = sentences[sent_i]
                    length = len(bart_tokenizer.encode(sent)[1:-1]) # ignore <s> and </s>
                    total_length += length
                    keep_idx.append(sent_i)
                else:
                    break

            keep_idx = sorted(keep_idx)

            filtered_sentences = [sentences[j] for j in keep_idx]
            filtered_transcription = " ".join(filtered_sentences)

        with open(out_path, "w") as f:
            f.write(filtered_transcription)

        print("write:", out_path)


def combine():
    with open(DATA_PATH, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    for i in tqdm(range(len(podcasts))):
        out_path = "data/podcast/oracle-padrand//decode{}/{}_filtered_transcription.txt".format(MAX_BART_LEN, PODCAST_SET, i)
        with open(out_path, 'r') as f:
            x = f.read()
        podcasts[i].transcription = x

    save_filtered_data_path = "data/podcast/oracle-padrand/podcast_set{}.bin".format(PODCAST_SET)
    with open(save_filtered_data_path, "wb") as f:
        pickle.dump(podcasts, f)

if __name__ == "__main__":
    # combine()

    if(len(sys.argv) == 2):
        start_id = int(sys.argv[1])
        end_id   = start_id + 250 # 10000 / 100 = 100
        if end_id > 10000: end_idx = 10000
        filtering_data(start_id, end_id)
    elif(len(sys.argv) == 3):
        start_id = int(sys.argv[1])
        end_id   = int(sys.argv[2])
        filtering_data(start_id, end_id)
    else:
        print("Usage: python oracle_select_pad_podcast.py start_id end_id")
        raise Exception("argv error")
