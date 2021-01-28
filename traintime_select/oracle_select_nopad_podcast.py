import os
import sys
import pickle
import random
import torch
import numpy as np
from nltk import tokenize
from tqdm import tqdm

from data.loader import BartBatcher, load_podcast_data
from data.processor import PodcastEpisode

from transformers import BartTokenizer

from rouge import Rouge
rouge_pltrdy = Rouge()


PODCAST_SET  = 9
DATA_PATH      = "../podcast_sum0/lib/data/podcast_set{}.bin".format(PODCAST_SET)
OUTDIR         = "data/podcast/oracle-padrand/"

print("PODCAST_SET:", PODCAST_SET)

def find_recall(sentences, reference):
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
    return scores

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

        sentences = tokenize.sent_tokenize(podcasts[i].transcription)
        reference = podcasts[i].description
        keep_idx = []

        scores = find_recall(sentences, reference)
        num_postive = sum(a > 0 for a in scores)
        rank = np.argsort(scores)[::-1][:num_postive] # only consider positive ones

        keep_idx = []
        total_length = 0
        for sent_i in rank:
            if total_length < 1024:
                sent = sentences[sent_i]
                length = len(bart_tokenizer.encode(sent)[1:-1]) # ignore <s> and </s>
                total_length += length
                keep_idx.append(sent_i)
            else:
                break

        # if found nothing, selecting the top3 longest sentences
        if len(keep_idx) == 0:
            sent_lengths = [len(tokenize.word_tokenize(ssent)) for ssent in sentences]
            keep_idx = np.argsort(sent_lengths)[::-1][:3].tolist()

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
        out_path = "data/podcast/oracle-nopad/decode{}/{}_filtered_transcription.txt".format(MAX_BART_LEN, PODCAST_SET, i)


        with open(out_path, 'r') as f:
            x = f.read()
        podcasts[i].transcription = x

    save_filtered_data_path = "data/podcast/oracle-nopad/podcast_set{}.bin".format(MAX_BART_LEN, PODCAST_SET)
    with open(save_filtered_data_path, "wb") as f:
        pickle.dump(podcasts, f)

if __name__ == "__main__":
    # combine()

    if(len(sys.argv) == 2):
        start_id = int(sys.argv[1])
        end_id   = start_id + 10 # 10000 / 200 = 50
        if end_id > 1027: end_idx = 1027
        filtering_data(start_id, end_id)
    elif(len(sys.argv) == 3):
        start_id = int(sys.argv[1])
        end_id   = int(sys.argv[2])
        filtering_data(start_id, end_id)
    else:
        print("Usage: python filtering_data.py start_id end_id")
        raise Exception("argv error")
