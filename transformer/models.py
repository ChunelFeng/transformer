#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from transformer.layers import DecoderLayer, EncoderLayer
from transformer.modules import PositionalEncoding
from transformer.utils import padding_mask, subsequent_mask


class Encoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_len,
                 n_layers=6,
                 d_model=512,
                 d_emb=512,
                 d_hidden=1024,
                 n_heads=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 pad_id=0):
        super(Encoder, self).__init__()

        self.pos_enc = PositionalEncoding(
            max_len + 1, d_emb, padding_idx=pad_id)
        self.emb = nn.Embedding(vocab_size, d_emb, padding_idx=pad_id)
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                d_hidden=d_hidden,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout) for _ in range(n_layers)
        ])

    def forward(self, seq, pos):
        mask = padding_mask(seq, seq)

        embedded = self.emb(seq)
        embedded += self.pos_enc(pos)

        outputs, attns = [], []
        output = embedded
        for layer in self.layers:
            output, attn = layer(output, slf_attn_mask=mask)
            outputs += [output]
            attns += [attn]

        return outputs, attns


class Decoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_len,
                 n_layers=6,
                 d_model=512,
                 d_emb=512,
                 d_hidden=1024,
                 n_heads=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 pad_id=0):
        super(Decoder, self).__init__()

        self.pos_enc = PositionalEncoding(
            max_len + 1, d_emb, padding_idx=pad_id)
        self.emb = nn.Embedding(vocab_size, d_emb, padding_idx=pad_id)
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                d_hidden=d_hidden,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout) for _ in range(n_layers)
        ])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_outputs):
        slf_attn_mask = torch.gt(
            padding_mask(tgt_seq, tgt_seq) + subsequent_mask(tgt_seq), 0)
        inter_attn_mask = padding_mask(tgt_seq, src_seq)

        embedded = self.emb(tgt_seq)
        embedded += self.pos_enc(tgt_pos)

        outputs, slf_attns, inter_attns = [], [], []
        output = embedded
        for layer, enc_output in zip(self.layers, enc_outputs):
            output, slf_attn, inter_attn = layer(
                output,
                enc_output,
                slf_attn_mask=slf_attn_mask,
                inter_attn_mask=inter_attn_mask)
            outputs += [output]
            slf_attns += [slf_attn]
            inter_attns += [inter_attn]

        return outputs, slf_attns, inter_attns


class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_len,
                 n_layers=6,
                 d_model=512,
                 d_emb=512,
                 d_hidden=1024,
                 n_heads=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 pad_id=0):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            d_model=d_model,
            d_emb=d_emb,
            d_hidden=d_hidden,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            pad_id=pad_id)

        self.decoder = Decoder(
            tgt_vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            d_model=d_model,
            d_emb=d_emb,
            d_hidden=d_hidden,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            pad_id=pad_id)

        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def get_trainable_parameters(self):
        freezed_param_ids = (
            set(id(p) for p in self.encoder.pos_enc.parameters())
            | set(id(p) for p in self.decoder.pos_enc.parameters()))

        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_outputs, _ = self.encoder(src_seq, src_pos)
        dec_outputs, _, _ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_outputs)

        logit = self.proj(dec_outputs[-1])

        return logit.view(-1, logit.size(2))
