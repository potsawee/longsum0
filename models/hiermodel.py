import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EncoderDecoder(nn.Module):
    def __init__(self, args, device):
        super(EncoderDecoder, self).__init__()
        self.device = device


        # Encoder - Hierarchical GRU
        self.encoder = HierarchicalGRU(args['vocab_size'], args['embedding_dim'], args['rnn_hidden_size'],
                                       num_layers=args['num_layers_enc'], dropout=args['dropout'], device=device)

        # Decoder - GRU with attention mechanism
        self.decoder = DecoderGRU(args['vocab_size'], args['embedding_dim'], args['rnn_hidden_size'], args['rnn_hidden_size'],
                                       num_layers=args['num_layers_dec'], dropout=args['dropout'], device=device)

        self.param_init()

        self.to(device)

    def param_init(self):
        # Initialisation
        # zero out the bias term
        # don't zero out LayerNorm term e.g. transformer_encoder.layers.0.norm1.weight
        for name, p in self.encoder.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if 'bias' in name: nn.init.zeros_(p)
        for name, p in self.decoder.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if 'bias' in name: nn.init.zeros_(p)

    def forward(self, input, u_len, w_len, target):
        enc_output_dict = self.encoder(input, u_len, w_len)
        dec_output, attn_scores, u_attn_scores = self.decoder(target, enc_output_dict)

        # compute coverage
        # cov_scores = self.attn2cov(attn_scores)
        return dec_output, enc_output_dict['u_output'], attn_scores, u_attn_scores

        # FOR multiple GPU training --- cannot have scores_uw (size error)
        # dec_output = self.decoder(target, enc_output_dict)
        # return dec_output

class HierarchicalGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_size, num_layers, dropout, device):
        super(HierarchicalGRU, self).__init__()
        self.device          = device
        self.vocab_size      = vocab_size
        self.embedding_dim   = embedding_dim
        self.rnn_hidden_size = rnn_hidden_size

        # embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0)

        # word-level GRU layer: word-embeddings -> utterance representation
        # divide by 2 becuase bi-directional
        self.gru_wlevel = nn.GRU(input_size=self.embedding_dim, hidden_size=int(self.rnn_hidden_size/2), num_layers=num_layers,
                                bias=True, batch_first=True, dropout=dropout, bidirectional=True)

        # utterance-level GRU layer (with  binary gate)
        self.gru_ulevel = nn.GRU(input_size=self.rnn_hidden_size, hidden_size=int(self.rnn_hidden_size/2), num_layers=num_layers,
                                bias=True, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, input, u_len, w_len):
        # input => [batch_size, num_utterances, num_words]
        # embed => [batch_size, num_utterances, num_words, embedding_dim]
        # embed => [batch_size*num_utterances,  num_words, embedding_dim]

        batch_size     = input.size(0)
        num_utterances = input.size(1)
        num_words      = input.size(2)

        embed = self.embedding(input)
        embed = embed.view(batch_size*num_utterances, num_words, self.embedding_dim)

        # word-level GRU
        self.gru_wlevel.flatten_parameters()
        w_output, _ = self.gru_wlevel(embed)
        w_len = w_len.reshape(-1)

        # utterance-level GRU
        utt_input = torch.zeros((w_output.size(0), w_output.size(2)), dtype=torch.float).to(self.device)
        for idx, l in enumerate(w_len):
            utt_input[idx] = w_output[idx, l-1]
        utt_input = utt_input.view(batch_size, num_utterances, self.rnn_hidden_size)
        self.gru_ulevel.flatten_parameters()
        utt_output, _ = self.gru_ulevel(utt_input)

        # reshape the output at different levels
        # w_output => [batch_size, num_utt, num_words, 2*hidden]
        # u_output => [batch_size, num_utt, hidden]
        w_output = w_output.view(batch_size, num_utterances, num_words, -1)
        w_len    = w_len.view(batch_size, -1)
        w2_len   = [None for _ in range(batch_size)]
        for bn, _l in enumerate(u_len):
            w2_len[bn] = w_len[bn, :_l].sum().item()

        w2_output = torch.zeros((batch_size, max(w2_len), w_output.size(-1))).to(self.device)
        utt_indices = [[] for _ in range(batch_size)]
        for bn, l1 in enumerate(u_len):
            x = 0
            for j, l2 in enumerate(w_len[bn, :l1]):
                w2_output[bn, x:x+l2, :] = w_output[bn, j, :l2, :]
                x += l2.item()
                utt_indices[bn].append(x-1) # minus one!!
        encoder_output_dict = {
            'u_output': utt_output, 'u_len': u_len,
            'w_output': w2_output, 'w_len': w2_len, 'utt_indices': utt_indices
        }
        return encoder_output_dict

class DecoderGRU(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, vocab_size, embedding_dim, dec_hidden_size, mem_hidden_size,
                num_layers, dropout, device):
        super(DecoderGRU, self).__init__()
        self.device      = device
        self.vocab_size  = vocab_size
        self.dec_hidden_size = dec_hidden_size
        self.mem_hidden_size = mem_hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        self.rnn = nn.GRU(embedding_dim, dec_hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.attention_u = nn.Linear(mem_hidden_size, dec_hidden_size)
        self.attention_w = nn.Linear(mem_hidden_size, dec_hidden_size)

        self.output_layer = nn.Linear(dec_hidden_size+mem_hidden_size, vocab_size, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, target, encoder_output_dict, logsoftmax=True):
        u_output = encoder_output_dict['u_output']
        u_len    = encoder_output_dict['u_len']
        w_output = encoder_output_dict['w_output']
        w_len    = encoder_output_dict['w_len']

        utt_indices     = encoder_output_dict['utt_indices']

        batch_size = target.size(0)

        embed = self.embedding(target)
        # initial hidden state
        initial_h = torch.zeros((self.num_layers, batch_size, self.dec_hidden_size), dtype=torch.float).to(self.device)
        for bn, l in enumerate(u_len):
            initial_h[:,bn,:] = u_output[bn,l-1,:].unsqueeze(0)

        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(embed, initial_h)

        # attention mechanism LEVEL --- Utterance (u)
        scores_u = torch.bmm(rnn_output, self.attention_u(u_output).permute(0,2,1))
        for bn, l in enumerate(u_len):
            scores_u[bn,:,l:].fill_(float('-inf'))
        scores_u = F.log_softmax(scores_u, dim=-1)

        # attention mechanism LEVEL --- Word (w)
        scores_w = torch.bmm(rnn_output, self.attention_w(w_output).permute(0,2,1))
        for bn, l in enumerate(w_len):
            scores_w[bn,:,l:].fill_(float('-inf'))
        # scores_w = F.log_softmax(scores_w, dim=-1)
        scores_uw = torch.zeros(scores_w.shape).to(self.device)
        scores_uw.fill_(float('-inf')) # when doing log-addition

        # Utterance -> Word
        for bn in range(batch_size):
            idx1 = 0
            idx2 = 0
            end_indices = utt_indices[bn]
            start_indices = [0] + [a+1 for a in end_indices[:-1]]
            for i in range(len(utt_indices[bn])):
                i1 = start_indices[i]
                i2 = end_indices[i]+1 # python
                scores_uw[bn, :, i1:i2] = scores_u[bn, :, i].unsqueeze(-1) + F.log_softmax(scores_w[bn, :, i1:i2], dim=-1)

        scores_uw = torch.exp(scores_uw)
        context_vec = torch.bmm(scores_uw, w_output)

        dec_output = self.output_layer(torch.cat((context_vec, rnn_output), dim=-1))

        if logsoftmax:
            dec_output = self.logsoftmax(dec_output)

        return dec_output, scores_uw, torch.exp(scores_u)

        # FOR multiple GPU training --- cannot have scores_uw (size error)
        # return dec_output

    def forward_step(self, xt, ht, encoder_output_dict, d_prev=None, eu_prev=None, logsoftmax=True):
        u_output = encoder_output_dict['u_output']
        u_len    = encoder_output_dict['u_len']
        w_output = encoder_output_dict['w_output']
        w_len    = encoder_output_dict['w_len']

        utt_indices     = encoder_output_dict['utt_indices']

        batch_size = xt.size(0)

        xt = self.embedding(xt) # xt => [batch_size, 1, input_size]
                                # ht => [batch_size, num_layers, hidden_size]

        rnn_output, ht1  = self.rnn(xt, ht)

        # attention mechanism LEVEL --- Utterance (u)
        scores_u = torch.bmm(rnn_output, self.attention_u(u_output).permute(0,2,1))
        for bn, l in enumerate(u_len):
            scores_u[bn,:,l:].fill_(float('-inf'))
        # scores_u = F.log_softmax(scores_u, dim=-1)
        scores_u = F.softmax(scores_u, dim=-1)

        # attention mechanism LEVEL --- Word (w)
        scores_w = torch.bmm(rnn_output, self.attention_w(w_output).permute(0,2,1))
        for bn, l in enumerate(w_len):
            scores_w[bn,:,l:].fill_(float('-inf'))
        scores_uw = torch.zeros(scores_w.shape).to(self.device)
        scores_uw.fill_(float('-inf')) # when doing log-addition

        # Utterance -> Word
        for bn in range(batch_size):
            idx1 = 0
            idx2 = 0
            end_indices = utt_indices[bn]
            start_indices = [0] + [a+1 for a in end_indices[:-1]]
            for i in range(len(utt_indices[bn])):
                i1 = start_indices[i]
                i2 = end_indices[i]+1 # python
                scores_uw[bn, :, i1:i2] = scores_u[bn, :, i].unsqueeze(-1) * F.softmax(scores_w[bn, :, i1:i2], dim=-1)


        # scores_uw = torch.exp(scores_uw)
        context_vec = torch.bmm(scores_uw, w_output)
        dec_output = self.output_layer(torch.cat((context_vec, rnn_output), dim=-1))


        if logsoftmax:
            logsm_dec_output = self.logsoftmax(dec_output)
            return logsm_dec_output[:,-1,:], ht1, scores_uw, scores_u, dec_output[:,-1,:]

        else:
            return dec_output[:,-1,:], ht1, scores_uw, scores_u, dec_output[:,-1,:]

class EXTLabeller(nn.Module):
    def __init__(self, dropout, rnn_hidden_size, device):
        super(EXTLabeller, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(rnn_hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else: nn.init.zeros_(p)

        self.to(device)

    def forward(self, utt_output):
        x = self.dropout(utt_output)
        x = self.linear(x)

        return self.sigmoid(x).squeeze(-1)
