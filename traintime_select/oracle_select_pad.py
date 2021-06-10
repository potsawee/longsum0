import os
import sys
import pickle
import random
import argparse
import numpy as np
from nltk import tokenize
from tqdm import tqdm

sys.path.insert(0, os.getcwd()+'/data/') # to import modules in data
from podcast_processor import PodcastEpisode
from arxiv_processor import ResearchArticle
from transformers import BartTokenizer
from utils import get_rouge2recall_scores

def oracle_filtering_data(args):
    dataset    = args['dataset']
    start_id   = args['start_id']
    end_id     = args['end_id']
    output_dir = args['output_dir']
    oracle_type    = args['oracle_type']
    max_abssum_len = args['max_abssum_len']

    if dataset not in ['podcast', 'arxiv', 'pubmed']:
        raise ValueError("Dataset not exist: only |podcast|arxiv|pubmed|")
    if oracle_type not in ['padrand', 'padlead']:
        raise Exception("oracle_type must be padrand or padlead")

    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    datapath = args['datapath']
    print("datapath =", datapath)
    with open(datapath, 'rb') as f:
        data = pickle.load(f, encoding="bytes")
    print("len(data) = {}".format(len(data)))

    ids = [x for x in range(start_id, end_id)]
    if args['random_order']: random.shuffle(ids)

    for i in ids:
        # check if the file exist or not
        # DECODER_DIR = temp folder
        out_path = "{}/{}_select.txt".format(output_dir, i)
        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(i))
            continue

        if args['dataset'] == 'podcast':
            input_text = data[i].transcription
            sentences = tokenize.sent_tokenize(input_text)
            reference = data[i].description
        elif args['dataset'] == 'arxiv' or args['dataset'] == 'pubmed':
            input_text = " ".join(data[i].article_text)
            sentences  = data[i].article_text
            reference = " ".join(data[i].abstract_text)

        l1 = len(bart_tokenizer.encode(input_text, max_length=50000))

        if l1 < max_abssum_len:
            filtered_input_text = input_text

        else:
            keep_idx = []
            selection_score = get_rouge2recall_scores(sentences, reference, oracle_type) # numpy array (80,)
            rank = np.argsort(selection_score)[::-1]
            keep_idx = []
            total_length = 0
            for sent_i in rank:
                if total_length < max_abssum_len:
                    sent = sentences[sent_i]
                    length = len(bart_tokenizer.encode(sent)[1:-1]) # ignore <s> and </s>
                    total_length += length
                    keep_idx.append(sent_i)
                else:
                    break
            keep_idx = sorted(keep_idx)
            filtered_sentences = [sentences[j] for j in keep_idx]
            filtered_input_text = " ".join(filtered_sentences)

        with open(out_path, "w") as f:
            f.write(filtered_input_text)
        print("write:", out_path)

def get_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # file paths
    parser.add_argument('--dataset',           type=str, required=True) # podcast | arxiv | pubmed
    parser.add_argument('--output_dir',   type=str, required=True)
    parser.add_argument('--datapath',     type=str, required=True)
    parser.add_argument('--oracle_type',  type=str, required=True) # padrand | padlead
    parser.add_argument('--max_abssum_len', type=int, required=True)

    parser.add_argument('--start_id',   type=int, required=True)
    parser.add_argument('--end_id',     type=int, required=True)
    parser.add_argument('--random_order', type="bool", nargs="?", const=True, default=False)

    return parser

if __name__ == "__main__":
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = get_arguments(parser)
    args = vars(parser.parse_args())
    oracle_filtering_data(args)
