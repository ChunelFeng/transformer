#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from transformer.modules import LayerNormalization, ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, d_k, d_v, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert n_heads * d_k == d_model, ('`n_heads` * `d_k` != `d_model`'
                                          ' ({} x {} != {})'.format(
                                              n_heads, d_k, d_model))

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_v))
        self.attn = ScaledDotProductAttention(d_k, attn_droput=dropout)

        self.proj = nn.Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

        nn.init.xavier_normal(self.w_q)
        nn.init.xavier_normal(self.w_k)
        nn.init.xavier_normal(self.w_v)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_heads = self.n_heads

        residual = q

        batch_size, len_q, d_model = q.size()
        batch_size, len_k, d_model = k.size()
        batch_size, len_v, d_model = v.size()

        q = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        k = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        v = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)

        q = torch.bmm(q, self.w_q).view(-1, len_q, d_k)
        k = torch.bmm(k, self.w_k).view(-1, len_k, d_k)
        v = torch.bmm(v, self.w_v).view(-1, len_v, d_v)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)

        o, attns = self.attn(q, k, v, attn_mask=attn_mask)
        o = torch.split(o, batch_size, dim=0)
        o = torch.cat(o, dim=-1)
        o = self.proj(o)
        o = self.dropout(o)

        attns = attns.view(n_heads, batch_size, len_q, len_k)

        return self.layer_norm(o + residual), attns


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_hidden=2048, d_model=512, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(d_model, d_hidden, 1)
        self.conv2 = nn.Conv1d(d_hidden, d_model, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x):
        o = self.conv1(torch.transpose(x, 1, 2))
        o = self.relu(o)
        o = torch.transpose(self.conv2(o), 1, 2)
        o = self.dropout(o)

        return self.layer_norm(o + x)
