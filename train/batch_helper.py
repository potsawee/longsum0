import os
import sys
sys.path.insert(0, os.getcwd()+'/data/') # to import modules in data

import pickle
import random
import torch
import numpy as np
from nltk import tokenize
from podcast_processor import PodcastEpisode
from arxiv_processor import ResearchArticle
from create_extractive_label import PodcastEpisodeXtra, ResearchArticleXtra


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

def load_podcast_data_xtra(dir, sets):
    """
    For training Hierarchical Model (with extractive labels)
    """
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "{}/podcast_ext1024_set{}.bin".format(dir, i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

# --------- arXiv / PubMed --------- #
def load_articles(path):
    """
    this function can load either ResearchArticle or ResearchArticleXtra
    """
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
        print("Shuffle data")
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

class HierPodcastBatcher(PodcastBatcher):
    def __init__(self, tokenizer, config, podcasts, torch_device):
        super().__init__(tokenizer, config, config['summary_length'], podcasts, torch_device)

        self.num_utterances = config['num_utterances']
        self.num_words      = config['num_words']
        self.summary_length = config['summary_length']

    # Override
    def get_a_batch(self, batch_size):
        """
        return input, u_len, w_len, target, tgt_len, ext_label
        """
        input = np.zeros((batch_size, self.num_utterances, self.num_words), dtype=np.long)
        u_len = np.zeros((batch_size), dtype=np.long)
        w_len = np.zeros((batch_size, self.num_utterances), dtype=np.long)
        ext_target = np.zeros((batch_size, self.num_utterances), dtype=np.float32)

        target  = np.zeros((batch_size, self.summary_length), dtype=np.long)
        target.fill(103) # in BERT 103 is [MASK] --- I should've used 0, which is pad_token
        tgt_len = np.zeros((batch_size), dtype=np.int)

        batch_count = 0
        while batch_count < batch_size:
            if not self.is_podcast_good(self.cur_id):
                self.increment_data_id()
                continue

            # ENCODER
            sentences = self.podcasts[self.cur_id].tran_split
            num_sentences = self.podcasts[self.cur_id].num_sent
            if num_sentences > self.num_utterances:
                num_sentences = self.num_utterances
                sentences = sentences[:self.num_utterances]
            u_len[batch_count] = num_sentences

            for j, sent in enumerate(sentences):
                token_ids = self.tokenizer.encode(sent.lower(), add_special_tokens=False)
                utt_len = len(token_ids)
                if utt_len > self.num_words:
                    utt_len = self.num_words
                    token_ids = token_ids[:self.num_words]
                input[batch_count,j,:utt_len] = token_ids
                w_len[batch_count,j] = utt_len

            # Extractive Sum Label
            positive_postions = [pi for pi in self.podcasts[self.cur_id].ext_label if pi < self.num_utterances]
            ext_target[batch_count][positive_postions] = 1.0

            # DECODER
            description   = self.podcasts[self.cur_id].description.lower()
            concat_tokens = [101]
            sentences = tokenize.sent_tokenize(description)
            for j, sent in enumerate(sentences):
                token_ids = self.tokenizer.encode(sent, add_special_tokens=False)
                concat_tokens.extend(token_ids)
                concat_tokens.extend([102]) # [SEP]
            tl = len(concat_tokens)
            if tl > self.summary_length:
                concat_tokens = concat_tokens[:self.summary_length]
                tl = self.summary_length
            target[batch_count, :tl] = concat_tokens
            tgt_len[batch_count] = tl

            # increment
            self.increment_data_id()
            batch_count += 1

        u_len_max = u_len.max()
        w_len_max = w_len.max()

        input = torch.from_numpy(input[:, :u_len_max, :w_len_max]).to(self.device)
        u_len = torch.from_numpy(u_len).to(self.device)
        w_len = torch.from_numpy(w_len[:, :u_len_max]).to(self.device)
        target = torch.from_numpy(target).to(self.device)
        ext_target = torch.from_numpy(ext_target[:, :u_len_max]).to(self.device)


        return input, u_len, w_len, target, tgt_len, ext_target



class HierArticleBatcher(ArticleBatcher):
    def __init__(self, tokenizer, config, articles, torch_device):
        super().__init__(tokenizer, config, config['summary_length'], articles, torch_device)

        self.num_utterances = config['num_utterances']
        self.num_words      = config['num_words']
        self.summary_length = config['summary_length']

    # Override
    def get_a_batch(self, batch_size):
        """
        return input, u_len, w_len, target, tgt_len, ext_label
        """
        input = np.zeros((batch_size, self.num_utterances, self.num_words), dtype=np.long)
        u_len = np.zeros((batch_size), dtype=np.long)
        w_len = np.zeros((batch_size, self.num_utterances), dtype=np.long)
        ext_target = np.zeros((batch_size, self.num_utterances), dtype=np.float32)

        target  = np.zeros((batch_size, self.summary_length), dtype=np.long)
        target.fill(103) # in BERT 103 is [MASK] --- I should've used 0, which is pad_token
        tgt_len = np.zeros((batch_size), dtype=np.int)

        batch_count = 0
        while batch_count < batch_size:
            # ENCODER
            sentences = self.articles[self.cur_id].article_text
            num_sentences = len(self.articles[self.cur_id].article_text)
            if num_sentences > self.num_utterances:
                num_sentences = self.num_utterances
                sentences = sentences[:self.num_utterances]
            u_len[batch_count] = num_sentences

            for j, sent in enumerate(sentences):
                token_ids = self.tokenizer.encode(sent.lower(), add_special_tokens=False, max_length=50000)
                utt_len = len(token_ids)
                if utt_len > self.num_words:
                    utt_len = self.num_words
                    token_ids = token_ids[:self.num_words]
                input[batch_count,j,:utt_len] = token_ids
                w_len[batch_count,j] = utt_len

            # Extractive Sum Label
            positive_postions = [pi for pi in self.articles[self.cur_id].ext_label if pi < self.num_utterances]
            ext_target[batch_count][positive_postions] = 1.0

            # DECODER
            description   =  " ".join(self.articles[self.cur_id].abstract_text).lower()
            concat_tokens = [101]
            sentences = tokenize.sent_tokenize(description)
            for j, sent in enumerate(sentences):
                token_ids = self.tokenizer.encode(sent, add_special_tokens=False, max_length=50000)
                concat_tokens.extend(token_ids)
                concat_tokens.extend([102]) # [SEP]
            tl = len(concat_tokens)
            if tl > self.summary_length:
                concat_tokens = concat_tokens[:self.summary_length]
                tl = self.summary_length
            target[batch_count, :tl] = concat_tokens
            tgt_len[batch_count] = tl

            # increment
            self.increment_data_id()
            batch_count += 1

        u_len_max = u_len.max()
        w_len_max = w_len.max()

        input = torch.from_numpy(input[:, :u_len_max, :w_len_max]).to(self.device)
        u_len = torch.from_numpy(u_len).to(self.device)
        w_len = torch.from_numpy(w_len[:, :u_len_max]).to(self.device)
        target = torch.from_numpy(target).to(self.device)
        ext_target = torch.from_numpy(ext_target[:, :u_len_max]).to(self.device)

        return input, u_len, w_len, target, tgt_len, ext_target
