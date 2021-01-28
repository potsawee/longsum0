import os
import sys
sys.path.insert(0, os.getcwd()+'/models/') # to import modules in models

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BartTokenizer, BartForConditionalGeneration
from localattn import LoBART
from pytorch_memlab import MemReporter



def mem_inspect(localattn=False, X=None, Y=None, W=None, B=None):
    if torch.cuda.is_available():
        torch_device = 'cuda'
    else:
        raise Exception("CUDA DEVICE is not available")

    # X = source len
    # Y = target len
    # W = window width
    print("X =", X)
    print("Y =", Y)
    print("W =", W)
    print("B =", B)

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
    reporter = MemReporter(model)

    # -------------- ARGV ------------ #
    batch_size     = B
    max_input_len  = X
    max_target_len = Y
    input_ids             = torch.zeros((batch_size, max_input_len), dtype=torch.long).cuda()
    attention_mask        = torch.ones((batch_size, max_input_len), dtype=torch.long).cuda()
    target_ids            = torch.zeros((batch_size, max_target_len), dtype=torch.long).cuda()
    target_attention_mask = torch.ones((batch_size, max_target_len), dtype=torch.long).cuda()

    # BART forward
    x = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=target_ids,decoder_attention_mask=target_attention_mask)
    lm_logits = x[0]
    loss = criterion(lm_logits.view(-1, bart_config.vocab_size), target_ids.view(-1)).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    x = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=target_ids,decoder_attention_mask=target_attention_mask)
    lm_logits = x[0]
    loss = criterion(lm_logits.view(-1, bart_config.vocab_size), target_ids.view(-1)).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    x = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=target_ids,decoder_attention_mask=target_attention_mask)
    lm_logits = x[0]
    loss = criterion(lm_logits.view(-1, bart_config.vocab_size), target_ids.view(-1)).mean()
    reporter.report(device=torch.device(0))


if __name__ == "__main__":
    """
        Inspect the peak memory in training BART/LoBART
        args:
            - localattn: bool (True | False)
            - X = max. input length
            - Y = max. target length
            - W = local attention window width (for localattn=True)
            - B = batch size
        memory = model_optim_mem + activation_mem*batch_size

        see the result in "The allocated memory on cuda:0: ...GiB"

    """
    if (len(sys.argv) == 6):
        localattn = bool(sys.argv[1])
        X = int(sys.argv[2])
        Y = int(sys.argv[3])
        W = int(sys.argv[4])
        B = int(sys.argv[5])
        mem_inspect(localattn=localattn, X=X, Y=Y, W=W, B=B)
    else:
        print("usage: python mem_inspect.py localattn X Y W B")
#
