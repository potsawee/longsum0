import os
import sys
sys.path.insert(0, os.getcwd()+'/models/') # to import modules in models

import time
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BartTokenizer, BartForConditionalGeneration
from localattn import LoBART
from transformers.modeling_bart import BartForConditionalGeneration, BartConfig, LearnedPositionalEmbedding


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def speed_inspect(localattn=False, X=None, Y=None, W=None, B=None, num_iterations=None):

    # X = source len
    # Y = target len
    # W = window width
    print("X =", X)
    print("Y =", Y)
    print("W =", W)
    print("B =", B)
    print("num_iterations =", num_iterations)

    xspan = 4  # this allows max. = 1024*xspan
    if 1024*xspan < X:
        raise ValueError("1024*xspan < X. change xspan to higher value")

    # Model & Optimizer
    if not localattn:
        model  = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    else:
        window_width = W
        attention_window = [window_width] * 12 # different window size for each layer can be defined here too!
        model = LoBART.from_pretrained('facebook/bart-large-cnn')
        model.swap_fullattn_to_localattn(attention_window=attention_window)
        model.expand_learned_embed_positions(multiple=xspan, cut=xspan*2)


    bart_config = model.config
    if torch_device == 'cuda': model.cuda()
    print("#parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()

    # Criterion
    criterion = nn.CrossEntropyLoss(reduction='none') # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

    model.train()

    # -------------- ARGV ------------ #
    batch_size     = B
    max_input_len  = X
    max_target_len = Y
    input_ids             = torch.zeros((batch_size, max_input_len), dtype=torch.long).cuda()
    attention_mask        = torch.ones((batch_size, max_input_len), dtype=torch.long).cuda()
    target_ids            = torch.zeros((batch_size, max_target_len), dtype=torch.long).cuda()
    target_attention_mask = torch.ones((batch_size, max_target_len), dtype=torch.long).cuda()


    # BART forward
    start = time.time()
    for it in range(num_iterations):
        x = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=target_ids,decoder_attention_mask=target_attention_mask)
        lm_logits = x[0]
        loss = criterion(lm_logits.view(-1, bart_config.vocab_size), target_ids.view(-1)).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("===:",it)

    end = time.time()
    print("-------------------------------------------")
    print("num_iterations:", num_iterations)
    print("time:", end - start)

if __name__ == "__main__":
    if (len(sys.argv) == 7):
        localattn = bool(sys.argv[1])
        X = int(sys.argv[2])
        Y = int(sys.argv[3])
        W = int(sys.argv[4])
        B = int(sys.argv[5])
        num_iterations = int(sys.argv[6])
        speed_inspect(localattn=localattn, X=X, Y=Y, W=W, B=B, num_iterations=num_iterations)
    else:
        print("usage: python speed_inspect.py localattn X Y W B num_iterations")
#
