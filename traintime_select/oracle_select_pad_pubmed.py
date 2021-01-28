import os
import sys
import pickle
import random
import numpy as np
from tqdm import tqdm

from data.loader import load_pubmed
from data.processing import ResearchArticle

from transformers import BartTokenizer

from rouge import Rouge
rouge_pltrdy = Rouge()

PUBMED_DATA_DIR = "data/pubmed"
OUTDIR         = "data/pubmed/oracle-padrand/4110/"
DATA_SET        = "train" # train | val | test
MAX_BART_LEN    = 4110
ORC_TYPE       = 'padrand' # padrand | padlead

print("DATA_SET:", DATA_SET)
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

        # from dump_original
        dumped_path = "data/pubmed/list_{}/{}_original_input.bin".format(DATA_SET, i)
        with open(dumped_path, 'rb') as f:
            x = pickle.load(f, encoding="bytes")

        original_article_text_list = x[0]
        abstract_text = x[1]

        article_text = " ".join(original_article_text_list)
        try:
            l1 = len(bart_tokenizer.encode(article_text, max_length=500000))
        except IndexError:
            l1 = 0

        if l1 < MAX_BART_LEN:
            article_text_list = original_article_text_list

        else:
            sentences = original_article_text_list
            reference = " ".join(abstract_text)

            keep_idx = []
            selection_score = get_rouge2recall_scores(sentences, reference) # numpy array (80,)
            rank = np.argsort(selection_score)[::-1]
            keep_idx = []
            total_length = 0
            for sent_i in rank:
                if total_length < MAX_BART_LEN:
                    sent = sentences[sent_i]
                    length = len(bart_tokenizer.encode(sent, add_special_tokens=False)) # ignore <s> and </s>
                    total_length += length
                    keep_idx.append(sent_i)
                else:
                    break

            keep_idx = sorted(keep_idx)

            article_text_list = [sentences[j] for j in keep_idx]
        with open(out_path, "wb") as f:
            pickle.dump(article_text_list, f)
        print("write:", out_path)


def combine():
    load_path = "{}/pubmed_{}.pk.bin".format(PUBMED_DATA_DIR, DATA_SET)
    with open(load_path, 'rb') as f:
        articles = pickle.load(f, encoding="bytes")
    print("len(articles) = {}".format(len(articles)))

    for i in tqdm(range(len(articles))):
        out_path = "data/pubmed/oracle-padrand//decode{}/{}_filtered_transcription.txt".format(MAX_BART_LEN, DATA_SET, i)

        try:
            with open(out_path, 'rb') as f:
                x = pickle.load(f, encoding="bytes")
            articles[i].article_text = x
        except:
            print("id: {} --- error".format(i))

    save_filtered_data_path = "data/pubmed/oracle-padrand/pubmed_set{}.bin".format(DATA_SET)
    with open(save_filtered_data_path, "wb") as f:
        pickle.dump(articles, f)

def dump_original(start_id, end_id):
    load_path = "{}/pubmed_{}.pk.bin".format(PUBMED_DATA_DIR, DATA_SET)
    with open(load_path, 'rb') as f:
        articles = pickle.load(f, encoding="bytes")
    print("len(articles) = {}".format(len(articles)))

    ids = [x for x in range(start_id, end_id)]
    # random.shuffle(ids)

    for i in ids:
        # check if the file exist or not
        # DECODER_DIR = temp folder
        out_path = "data/pubmed/list_{}/{}_original_input.bin".format(DATA_SET, i)
        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(i))
            continue

        with open(out_path, "wb") as f:
            pickle.dump((articles[i].article_text, articles[i].abstract_text), f)
        print("write:", out_path)

if __name__ == "__main__":
    # combine()

    if(len(sys.argv) == 2):
        start_id = int(sys.argv[1])
        end_id   = start_id + 500
        if end_id > 119924: end_idx = 119924 # train=119924, val=6633, test=6658
        filtering_data(start_id, end_id)
    elif(len(sys.argv) == 3):
        start_id = int(sys.argv[1])
        end_id   = int(sys.argv[2])
        filtering_data(start_id, end_id)
    else:
        print("Usage: python filtering_data.py start_id end_id")
        raise Exception("argv error")
