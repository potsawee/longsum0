import os
import sys
sys.path.insert(0, os.getcwd()+'/data/') # to import modules in data
sys.path.insert(0, os.getcwd()+'/models/') # to import modules in models

import random
from datetime import datetime
from collections import OrderedDict

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# HuggingFace
from transformers import BartTokenizer, BartForConditionalGeneration

# This project
from utils import parse_config, print_config, adjust_lr
from batch_helper import load_podcast_data, load_articles, PodcastBatcher, ArticleBatcher
from podcast_processor import PodcastEpisode
from arxiv_processor import ResearchArticle
from localattn import LoBART

def run_training(config_path):
    # Load Config
    config = parse_config("config", config_path)
    print_config(config)

    # uses GPU in training or not
    if torch.cuda.is_available() and config['use_gpu']: torch_device = 'cuda'
    else: torch_device = 'cpu'

    bart_tokenizer = BartTokenizer.from_pretrained(config['bart_tokenizer'])
    if config['selfattn'] == 'full':
        bart_model  = BartForConditionalGeneration.from_pretrained(config['bart_weights'])
    elif config['selfattn'] == 'local':
        window_width = config['window_width']
        xspan        = config['multiple_input_span']
        attention_window = [window_width] * 12 # different window size for each layer can be defined here too!
        bart_model = LoBART.from_pretrained(config['bart_weights'])
        bart_model.swap_fullattn_to_localattn(attention_window=attention_window)
        bart_model.expand_learned_embed_positions(multiple=xspan, cut=xspan*2)
    else:
        raise ValueError("selfattn: full (BART) | local (LoBART)")
    bart_config = bart_model.config

    # print out model details for future reference
    print(bart_model)
    print(bart_config)
    print("#parameters:", sum(p.numel() for p in bart_model.parameters() if p.requires_grad))
    if torch_device == 'cuda': bart_model.cuda()

    # Optimizer --- currently only support Adam
    if config['optimizer'] == 'adam':
        # lr here doesn't matter as it will be changed by .adjust_lr()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, bart_model.parameters()), lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
        optimizer.zero_grad()
    else:
        raise ValueError("Current version only supports Adam")


    # Data ---- podcast | arxiv | pubmed
    if config['dataset'] == 'podcast':
        # train_data  = load_podcast_data(config['data_dir'], sets=-1)   # -1 = training set, -1 means set0,..,set9 (excluding 10)
        train_data  = load_podcast_data(config['data_dir'], sets=[10])   # -1 = training set, -1 means set0,..,set9 (excluding 10)
        val_data    = load_podcast_data(config['data_dir'], sets=[10]) # 10 = valid set
        batcher     = PodcastBatcher(bart_tokenizer, bart_config, config['max_target_len'], train_data, torch_device)
        val_batcher = PodcastBatcher(bart_tokenizer, bart_config, config['max_target_len'], val_data, torch_device)
    elif config['dataset'] == 'arxiv':
        # train_data  = load_articles("{}/arxiv_train.pk.bin".format(config['data_dir']))
        train_data  = load_articles("{}/arxiv_val.pk.bin".format(config['data_dir']))
        val_data    = load_articles("{}/arxiv_val.pk.bin".format(config['data_dir']))
        batcher     = ArticleBatcher(bart_tokenizer, bart_config, config['max_target_len'], train_data, torch_device)
        val_batcher = ArticleBatcher(bart_tokenizer, bart_config, config['max_target_len'], val_data, torch_device)
    elif config['dataset'] == 'pubmed':
        train_data  = load_articles("{}/pubmed_train.pk.bin".format(config['data_dir']))
        val_data    = load_articles("{}/pubmed_val.pk.bin".format(config['data_dir']))
        batcher     = ArticleBatcher(bart_tokenizer, bart_config, config['max_target_len'], train_data, torch_device)
        val_batcher = ArticleBatcher(bart_tokenizer, bart_config, config['max_target_len'], val_data, torch_device)
    else:
        raise ValueError("Dataset not exist: only |podcast|arxiv|pubmed|")

    # Criterion
    criterion = nn.CrossEntropyLoss(reduction='none') # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

    training_step  = 0
    best_val_loss  = 9e9
    stop_counter   = 0
    batch_size     = config['batch_size']
    lr0            = config['lr0']
    warmup         = config['warmup']
    gradient_accum = config['gradient_accum']
    valid_step     = config['valid_step']
    total_step     = config['total_step']
    early_stop     = config['early_stop']
    random_seed    = config['random_seed']

    # Randomness
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # shuffle data
    batcher.shuffle_data()
    bart_model.train() # by default, it's not training!!!

    while training_step < total_step:

        input_ids, attention_mask, target_ids, target_attention_mask = batcher.get_a_batch(batch_size=batch_size, pad_to_max_length=False)
        shifted_target_ids, shifted_target_attention_mask = batcher.shifted_target_left(target_ids, target_attention_mask)

        # BART forward
        x = bart_model(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=target_ids, decoder_attention_mask=target_attention_mask,
        )
        # x[0] # decoder output
        # x[1] # encoder output
        lm_logits = x[0]

        loss = criterion(lm_logits.view(-1, bart_config.vocab_size), shifted_target_ids.view(-1))
        shifted_target_attention_mask = shifted_target_attention_mask.view(-1)
        loss = (loss * shifted_target_attention_mask).sum() / shifted_target_attention_mask.sum()
        loss.backward()

        if training_step % gradient_accum == 0:
            adjust_lr(optimizer, training_step, lr0, warmup)
            optimizer.step()
            optimizer.zero_grad()

        if training_step % 10 == 0:
            print("[{}] step {}/{}: loss = {:.5f}".format(str(datetime.now()), training_step, total_step, loss))
            sys.stdout.flush()

        if training_step % valid_step == 0 and training_step > 0:
            bart_model.eval()
            with torch.no_grad():
                valid_loss = validation(bart_model, bart_config, val_batcher, batch_size)
            print("Valid Loss = {:.5f}".format(valid_loss))
            bart_model.train()
            if valid_loss < best_val_loss:
                stop_counter = 0
                best_val_loss = valid_loss
                print("Model improved".format(stop_counter))
            else:
                stop_counter += 1
                print("Model not improved #{}".format(stop_counter))
                if stop_counter == early_stop:
                    print("Stop training!")
                    return

            state = {
                'training_step': training_step,
                'model': bart_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            savepath = "{}/{}-step{}.pt".format(config['save_dir'], config['model_name'], training_step)
            torch.save(state, savepath)
            print("Saved at {}".format(savepath))

        training_step += 1
    print("Finish training abstractive summarizer")


def validation(bart, bart_config, val_batcher, batch_size):
    print("start validating")
    criterion = nn.CrossEntropyLoss(reduction='none')
    sum_loss = 0
    sum_token = 0
    while val_batcher.epoch_counter < 1:
    # for i in range(5):
        input_ids, attention_mask, target_ids, target_attention_mask = val_batcher.get_a_batch(batch_size=batch_size, pad_to_max_length=False)

        shifted_target_ids, shifted_target_attention_mask = val_batcher.shifted_target_left(target_ids, target_attention_mask)
        x = bart(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=target_ids, decoder_attention_mask=target_attention_mask,
        )
        lm_logits = x[0]
        loss = criterion(lm_logits.view(-1, bart_config.vocab_size), shifted_target_ids.view(-1))
        shifted_target_attention_mask = shifted_target_attention_mask.view(-1)
        sum_loss += (loss * shifted_target_attention_mask).sum().item()
        sum_token += shifted_target_attention_mask.sum().item()
        print("#", end="")
        sys.stdout.flush()

    print()
    val_batcher.epoch_counter = 0
    val_batcher.cur_id = 0
    print("finish validating")
    return sum_loss / sum_token


if __name__ == "__main__":
    if(len(sys.argv) == 2):
        run_training(sys.argv[1])
    else:
        print("Usage: python train_bart.py config_path")
