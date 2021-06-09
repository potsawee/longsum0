import os
import sys
sys.path.insert(0, os.getcwd()+'/data/') # to import modules in data
import pickle
import argparse
from tqdm import tqdm
from podcast_processor import PodcastEpisode
from arxiv_processor import ResearchArticle

def inference_hiermodel_combine(args):
    dataset     = args['dataset']
    decoded_dir = args['decoded_dir']
    original_datapath = args['original_datapath']
    filtered_datapath = args['filtered_datapath']

    print("original_datapath =", original_datapath)
    with open(original_datapath, 'rb') as f:
        data = pickle.load(f, encoding="bytes")
    print("len(data) = {}".format(len(data)))

    for id in tqdm(range(len(data))):
        with open("{}/{}_decoded.pk.bin".format(decoded_dir, id), 'rb') as f:
            filtered_sentences = pickle.load(f, encoding="bytes")
        if args['dataset'] == 'podcast':
            data[id].transcription = " ".join(filtered_sentences)
        elif args['dataset'] == 'arxiv' or args['dataset'] == 'pubmed':
            data[id].article_text = filtered_sentences
        else:
            raise ValueError("Dataset not exist: only |podcast|arxiv|pubmed|")

    with open(filtered_datapath, "wb") as f:
        pickle.dump(data, f)
    print("dumped:", filtered_datapath)

def get_decode_arguments(parser):
    '''Arguments for decoding'''
    # parser.register("type", "bool", lambda v: v.lower() == "true")
    # file paths
    parser.add_argument('--dataset',           type=str, required=True)
    parser.add_argument('--decoded_dir',       type=str, required=True)
    parser.add_argument('--original_datapath', type=str, required=True)
    parser.add_argument('--filtered_datapath', type=str, required=True)

    return parser

if __name__ == "__main__":
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = get_decode_arguments(parser)
    args = vars(parser.parse_args())
    inference_hiermodel_combine(args)
