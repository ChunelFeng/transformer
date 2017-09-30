#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from transformer.sublayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(
            n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, dropout=dropout)
        self.fc = PositionwiseFeedForward(
            d_hidden=d_hidden, d_model=d_model, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        output, slf_attn = self.attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        output = self.fc(output)

        return output, slf_attn


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(
            n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, dropout=dropout)
        self.inter_attn = MultiHeadAttention(
            n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, dropout=dropout)
        self.fc = PositionwiseFeedForward(
            d_hidden=d_hidden, d_model=d_model, dropout=dropout)

    def forward(self,
                dec_input,
                enc_output,
                slf_attn_mask=None,
                inter_attn_mask=None):

        output, slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        output, inter_attn = self.inter_attn(
            output, enc_output, enc_output, attn_mask=inter_attn_mask)
        output = self.fc(output)

        return output, slf_attn, inter_attn
