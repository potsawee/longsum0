import os
import sys
sys.path.insert(0, os.getcwd()+'/data/') # to import modules in data
sys.path.insert(0, os.getcwd()+'/models/') # to import modules in models

import numpy as np
import pickle
import random
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer

from utils import parse_config, print_config, adjust_lr
from batch_helper import load_podcast_data_xtra, load_articles, HierPodcastBatcher, HierArticleBatcher
from podcast_processor import PodcastEpisode
from arxiv_processor import ResearchArticle
from create_extractive_label import PodcastEpisodeXtra, ResearchArticleXtra
from hiermodel import EncoderDecoder, EXTLabeller

def run_training(config_path):
    # Load Config
    config = parse_config("config", config_path)
    print_config(config)

    # uses GPU in training or not
    if torch.cuda.is_available() and config['use_gpu']: torch_device = 'cuda'
    else: torch_device = 'cpu'

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = EncoderDecoder(config, device=torch_device)
    print(model)
    ext_labeller = EXTLabeller(rnn_hidden_size=config['rnn_hidden_size'], dropout=config['dropout'], device=torch_device)
    print(ext_labeller)
    print("#enc-dec parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("#ext-labeller parameters:", sum(p.numel() for p in ext_labeller.parameters() if p.requires_grad))

    if torch_device == 'cuda':
        model.cuda()
        ext_labeller.cuda()

    # Optimizer --- currently only support Adam
    if config['optimizer'] == 'adam':
        # lr here doesn't matter as it will be changed by .adjust_lr()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
        optimizer.zero_grad()
        ext_optimizer = optim.Adam(ext_labeller.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
        ext_optimizer.zero_grad()
    else:
        raise ValueError("Current version only supports Adam")

    # Data ---- podcast | arxiv | pubmed
    if config['dataset'] == 'podcast':
        train_data  = load_podcast_data(config['data_dir'], sets=-1)   # -1 = training set, -1 means set0,..,set9 (excluding 10)
        val_data    = load_podcast_data_xtra(config['data_dir'], sets=[10]) # 10 = valid set
        batcher     = HierPodcastBatcher(bert_tokenizer, config, train_data, torch_device)
        val_batcher = HierPodcastBatcher(bert_tokenizer, config, val_data, torch_device)
    elif config['dataset'] == 'arxiv':
        train_data  = load_articles("{}/arxiv_train.pk.bin".format(config['data_dir']))
        val_data    = load_articles("{}/arxiv_val.pk.bin".format(config['data_dir']))
        batcher     = HierArticleBatcher(bert_tokenizer, config, train_data, torch_device)
        val_batcher = HierArticleBatcher(bert_tokenizer, config, val_data, torch_device)
    elif config['dataset'] == 'pubmed':
        train_data  = load_articles("{}/pubmed_train.pk.bin".format(config['data_dir']))
        val_data    = load_articles("{}/pubmed_val.pk.bin".format(config['data_dir']))
        batcher     = HierArticleBatcher(bert_tokenizer, config, train_data, torch_device)
        val_batcher = HierArticleBatcher(bert_tokenizer, config, val_data, torch_device)
    else:
        raise ValueError("Dataset not exist: only |podcast|arxiv|pubmed|")



    criterion = nn.NLLLoss(reduction='none')
    ext_criterion = nn.BCELoss(reduction='none')

    training_step  = 0
    best_val_loss  = 9e9
    stop_counter   = 0
    batch_size     = config['batch_size']
    gamma          = config['gamma']
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
    model.train() # by default, it's not training!!!

    while training_step < total_step:

            # get a batch
            input, u_len, w_len, target, tgt_len, ext_target = batcher.get_a_batch(batch_size)

            # decoder target
            decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, torch_device, mask_offset=False)
            decoder_target = decoder_target.view(-1)
            decoder_mask = decoder_mask.view(-1)

            # Forward pass
            decoder_output, enc_u_output, attn_scores, u_attn_scores = model(input, u_len, w_len, target)

            # Multitask Learning: Task 1 - Predicting targets
            loss1 = criterion(decoder_output.view(-1, config['vocab_size']), decoder_target)
            loss1 = (loss1 * decoder_mask).sum() / decoder_mask.sum()

            # Multitask Learning: Task 2 - Extractive Summarisation
            loss_utt_mask = length2mask(u_len, batch_size, u_len.max().item(), torch_device)
            ext_output = ext_labeller(enc_u_output).squeeze(-1)
            loss2 = ext_criterion(ext_output, ext_target)
            loss2 = (loss2 * loss_utt_mask).sum() / loss_utt_mask.sum()

            loss = (1-gamma)*loss1 + gamma*loss2
            loss.backward()

            if training_step % gradient_accum == 0:
                adjust_lr(optimizer, training_step, lr0, warmup)
                adjust_lr(ext_optimizer, training_step, lr0, warmup)
                optimizer.step()
                optimizer.zero_grad()
                ext_optimizer.step()
                ext_optimizer.zero_grad()

            if training_step % 10 == 0:
                print("[{}] step {}/{}: loss = {:.4f} | loss_pred = {:.4f} | loss_ext = {:.4f}".format(
                    str(datetime.now()), training_step, total_step, loss, loss1, loss2))
                sys.stdout.flush()

            # if training_step % 200 == 0:
            #     print("======================== GENERATED SUMMARY ========================")
            #     print(bert_tokenizer.decode(torch.argmax(decoder_output[0], dim=-1).cpu().numpy()[:tgt_len[0]]))
            #     print("======================== REFERENCE SUMMARY ========================")
            #     print(bert_tokenizer.decode(decoder_target.view(batch_size,config['summary_length'])[0,:tgt_len[0]].cpu().numpy()))

            if training_step % valid_step == 0 and training_step > 0:
                # ---------------- Evaluate the model on validation data ---------------- #
                print("Evaluating the model at training step {}".format(training_step))
                print("learning_rate = {}".format(optimizer.param_groups[0]['lr']))
                # switch to evaluation mode
                model.eval()
                ext_labeller.eval()
                with torch.no_grad():
                    valid_loss = evaluate(model, ext_labeller, gamma, val_batcher, batch_size, config, torch_device)
                print("valid_loss = {}".format(valid_loss))
                # switch to training mode
                model.train()
                ext_labeller.train()

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
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ext_labeller': ext_labeller.state_dict(),
                    'ext_optimizer': ext_optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }
                savepath = "{}/{}-step{}.pt".format(config['save_dir'], config['model_name'], training_step)
                torch.save(state, savepath)
                print("Saved at {}".format(savepath))

            training_step += 1

    print("End of training hierarchical RNN model")

def evaluate(model, ext_labeller, gamma, val_batcher, batch_size, config, torch_device):
    print("start validating")
    criterion = nn.NLLLoss(reduction='none')
    ext_criterion = nn.BCELoss(reduction='none')

    eval_total_loss1 = 0.0
    eval_total_loss2 = 0.0
    eval_total_tokens1 = 0
    eval_total_tokens2 = 0

    while val_batcher.epoch_counter < 1:
    # for i in range(5):
        input, u_len, w_len, target, tgt_len, ext_target = val_batcher.get_a_batch(batch_size)

        # decoder target
        decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, torch_device)
        decoder_target = decoder_target.view(-1)
        decoder_mask = decoder_mask.view(-1)
        decoder_output, enc_u_output, _, _ = model(input, u_len, w_len, target)

        loss1 = criterion(decoder_output.view(-1, config['vocab_size']), decoder_target)

        loss_utt_mask = length2mask(u_len, batch_size, u_len.max().item(), torch_device)
        ext_output = ext_labeller(enc_u_output).squeeze(-1)
        loss2 = ext_criterion(ext_output, ext_target)

        eval_total_loss1   += (loss1 * decoder_mask).sum().item()
        eval_total_loss2  += (loss2 * loss_utt_mask).sum().item()

        eval_total_tokens1 += decoder_mask.sum().item()
        eval_total_tokens2 += loss_utt_mask.sum().item()

        print("#", end="")
        sys.stdout.flush()
    print()
    avg_eval_loss1 = eval_total_loss1 / eval_total_tokens1
    avg_eval_loss2 = eval_total_loss2 / eval_total_tokens2
    val_batcher.epoch_counter = 0
    val_batcher.cur_id = 0
    print("finish validating")
    avg_eval_loss = (1-gamma)*avg_eval_loss1 + gamma*avg_eval_loss2
    return avg_eval_loss

def length2mask(length, batch_size, max_len, torch_device):
    mask = torch.zeros((batch_size, max_len), dtype=torch.float, device=torch_device)
    for bn in range(batch_size):
        l = length[bn].item()
        mask[bn,:l].fill_(1.0)
    return mask

def shift_decoder_target(target, tgt_len, torch_device, mask_offset=False):
    # MASK_TOKEN_ID = 103
    batch_size = target.size(0)
    max_len = target.size(1)
    dtype0  = target.dtype

    decoder_target = torch.zeros((batch_size, max_len), dtype=dtype0, device=torch_device)
    decoder_target[:,:-1] = target.clone().detach()[:,1:]
    # decoder_target[:,-1:] = 103 # MASK_TOKEN_ID = 103
    # decoder_target[:,-1:] = 0 # add padding id instead of MASK

    # mask for shifted decoder target
    decoder_mask = torch.zeros((batch_size, max_len), dtype=torch.float, device=torch_device)
    if mask_offset:
        offset = 10
        for bn, l in enumerate(tgt_len):
            # decoder_mask[bn,:l-1].fill_(1.0)
            # to accommodate like 10 more [MASK] [MASK] [MASK] [MASK],...
            if l-1+offset < max_len: decoder_mask[bn,:l-1+offset].fill_(1.0)
            else: decoder_mask[bn,:].fill_(1.0)
    else:
        for bn, l in enumerate(tgt_len):
            decoder_mask[bn,:l-1].fill_(1.0)

    return decoder_target, decoder_mask

if __name__ == "__main__":
    if(len(sys.argv) == 2):
        run_training(sys.argv[1])
    else:
        print("Usage: python train_hiermodel.py config_path")
