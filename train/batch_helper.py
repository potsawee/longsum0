import sys
# sys.path.insert(0, '../data/') # to import modules in data
import pickle
import random
import torch
import numpy as np

from podcast_processor import PodcastEpisode
from arxiv_processor import ResearchArticle


# --------- Spotify Podcast --------- #
# DEFINE PATH HERE:
BRASS_SET_PATH   = "data/spotify-podcasts/summarisation-task-brass-set/filtered-episode-ids.txt"
DEV150_SET_PATH  = "data/spotify-podcasts/no-audio/spotify-podcasts-2020/dev-set/150-episode-ids.txt"
def load_podcast_data(data_dir, sets):
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "{}/podcast_set{}.bin".format(data_dir, i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

def load_brass_set_ids():
    with open(BRASS_SET_PATH, 'r') as f:
        lines = f.readlines()
    ids = [None for _ in range(len(lines))]
    for i, line in enumerate(lines):
        id = int(line.strip())
        ids[i] = id
    return ids
def load_dev150_set_ids():
    with open(DEV150_SET_PATH, 'r') as f:
        lines = f.readlines()
    ids = [None for _ in range(len(lines))]
    for i, line in enumerate(lines):
        id = int(line.strip())
        ids[i] = id
    return ids

# --------- arXiv / PubMed --------- #
def load_articles(path):
    with open(path, 'rb') as f:
        articles = pickle.load(f, encoding="bytes")
    print("loaded:", path)
    return articles

class PodcastBatcher(object):
    def __init__(self, tokenizer, bart_config, max_target_len, podcasts, torch_device):
        self.cur_id    = 0
        self.epoch_counter = 0
        self.max_count = len(podcasts)
        self.device    = torch_device

        self.tokenizer = tokenizer
        self.podcasts  = podcasts
        self.config    = bart_config
        self.max_target_len = max_target_len
        self.brass_set_ids  = load_brass_set_ids()
        self.dev150_set_ids = load_dev150_set_ids()


    def shuffle_data(self):
        print("Suhffle data")
        random.shuffle(self.podcasts)
        return

    def is_podcast_good(self, id):
        podcast_id = self.podcasts[id].podcast_id
        if podcast_id in self.brass_set_ids and podcast_id not in self.dev150_set_ids and len(self.podcasts[id].description.split()) >= 5:
            return True
        else:
            return False

    def increment_data_id(self):
        self.cur_id += 1
        if self.cur_id == self.max_count:
            self.cur_id = 0
            self.shuffle_data()
            self.epoch_counter += 1
        return

    def get_a_batch(self, batch_size, pad_to_max_length=True):
        batch_count = 0
        inputs  = [None for _ in range(batch_size)]
        targets = [None for _ in range(batch_size)]

        while batch_count < batch_size:
            if not self.is_podcast_good(self.cur_id):
                self.increment_data_id()
                continue

            inputs[batch_count]  = self.podcasts[self.cur_id].transcription
            targets[batch_count] = self.podcasts[self.cur_id].description

            self.increment_data_id()
            batch_count += 1

        batch_encoded_inputs = self.tokenizer.batch_encode_plus(inputs,
            add_special_tokens=True, pad_to_max_length=True,
            max_length=self.config.max_position_embeddings, return_tensors='pt')

        input_ids      = batch_encoded_inputs['input_ids']
        attention_mask = batch_encoded_inputs['attention_mask']

        batch_encoded_targets = self.tokenizer.batch_encode_plus(targets,
            add_special_tokens=True, pad_to_max_length=pad_to_max_length,
            max_length=self.max_target_len, return_tensors='pt')

        target_ids = batch_encoded_targets['input_ids']
        target_attention_mask = batch_encoded_targets['attention_mask']

        if self.device == 'cuda':
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            target_ids = target_ids.to(self.device)
            target_attention_mask = target_attention_mask.to(self.device)

        return input_ids, attention_mask, target_ids, target_attention_mask

    def shifted_target_left(self, target_ids, target_attention_mask):
        # shifted LEFT
        shifted_target_ids = torch.zeros(target_ids.shape, dtype=target_ids.dtype)
        shifted_target_attention_mask = torch.zeros(target_attention_mask.shape, dtype=torch.float)
        shifted_target_ids[:,:-1] = target_ids.clone().detach()[:,1:]
        shifted_target_attention_mask[:,:-1] = target_attention_mask.clone().detach()[:,1:]

        if self.device == 'cuda':
            shifted_target_ids = shifted_target_ids.to(self.device)
            shifted_target_attention_mask = shifted_target_attention_mask.to(self.device)

        return shifted_target_ids, shifted_target_attention_mask

class ArticleBatcher(object):
    def __init__(self, tokenizer, bart_config, max_sum_len, articles, torch_device):
        self.cur_id    = 0
        self.epoch_counter = 0
        self.max_count = len(articles)
        self.device    = torch_device

        self.tokenizer = tokenizer
        self.articles  = articles
        self.config    = bart_config
        self.max_sum_len = max_sum_len

    def shuffle_data(self):
        print("Shuffle data...")
        random.shuffle(self.articles)
        return


    def increment_data_id(self):
        self.cur_id += 1
        if self.cur_id == self.max_count:
            self.cur_id = 0
            self.shuffle_data()
            self.epoch_counter += 1
        return

    def get_a_batch(self, batch_size, pad_to_max_length=True):
        batch_count = 0
        inputs  = [None for _ in range(batch_size)]
        targets = [None for _ in range(batch_size)]

        while batch_count < batch_size:
            inputs[batch_count]  = " ".join(self.articles[self.cur_id].article_text)
            targets[batch_count] = " ".join(self.articles[self.cur_id].abstract_text)

            self.increment_data_id()
            batch_count += 1

        batch_encoded_inputs = self.tokenizer.batch_encode_plus(inputs,
            add_special_tokens=True, pad_to_max_length=True,
            max_length=self.config.max_position_embeddings, return_tensors='pt')

        input_ids      = batch_encoded_inputs['input_ids']
        attention_mask = batch_encoded_inputs['attention_mask']

        batch_encoded_targets = self.tokenizer.batch_encode_plus(targets,
            add_special_tokens=True, pad_to_max_length=pad_to_max_length,
            max_length=self.max_sum_len, return_tensors='pt')

        # bos_token_id = 0
        target_ids = batch_encoded_targets['input_ids']
        target_attention_mask = batch_encoded_targets['attention_mask']

        if self.device == 'cuda':
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            target_ids = target_ids.to(self.device)
            target_attention_mask = target_attention_mask.to(self.device)

        return input_ids, attention_mask, target_ids, target_attention_mask

    def shifted_target_left(self, target_ids, target_attention_mask):
        # shifted LEFT
        shifted_target_ids = torch.zeros(target_ids.shape, dtype=target_ids.dtype)
        shifted_target_attention_mask = torch.zeros(target_attention_mask.shape, dtype=torch.float)
        shifted_target_ids[:,:-1] = target_ids.clone().detach()[:,1:]
        shifted_target_attention_mask[:,:-1] = target_attention_mask.clone().detach()[:,1:]

        if self.device == 'cuda':
            shifted_target_ids = shifted_target_ids.to(self.device)
            shifted_target_attention_mask = shifted_target_attention_mask.to(self.device)

        return shifted_target_ids, shifted_target_attention_mask
