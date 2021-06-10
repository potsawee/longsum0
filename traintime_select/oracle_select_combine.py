import os
import sys
sys.path.insert(0, os.getcwd()+'/data/') # to import modules in data
import pickle
import argparse
from tqdm import tqdm
from podcast_processor import PodcastEpisode
from arxiv_processor import ResearchArticle

def combine(args):
    dataset     = args['dataset']
    output_dir = args['output_dir']
    original_datapath = args['original_datapath']
    filtered_datapath = args['filtered_datapath']

    with open(original_datapath, 'rb') as f:
        data = pickle.load(f, encoding="bytes")
    print("len(data) = {}".format(len(data)))

    for i in tqdm(range(len(data))):
        out_path = "{}/{}_select.txt".format(output_dir, i)
        with open(out_path, 'r') as f:
            x = f.read()
        if args['dataset'] == 'podcast':
            data[i].transcription = x
        elif args['dataset'] == 'arxiv' or args['dataset'] == 'pubmed':
            pass # TODO
        else:
            raise ValueError("Dataset not exist: only |podcast|arxiv|pubmed|")

    with open(filtered_datapath, "wb") as f:
        pickle.dump(data, f)

def get_combine_arguments(parser):
    # parser.register("type", "bool", lambda v: v.lower() == "true")
    # file paths
    parser.add_argument('--dataset',           type=str, required=True)
    parser.add_argument('--output_dir',       type=str, required=True)
    parser.add_argument('--original_datapath', type=str, required=True)
    parser.add_argument('--filtered_datapath', type=str, required=True)

    return parser

if __name__ == "__main__":
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = get_combine_arguments(parser)
    args = vars(parser.parse_args())
    combine(args)
