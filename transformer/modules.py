#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn


class Bottle(nn.Module):

    def forward(self, x):
        if len(x.size()) <= 2:
            return super(Bottle, self).forward(x)

        size = x.size()[:2]
        o = super(Bottle, self).forward(x.view(size[0] * size[1], -1))

        return o.view(size[0], size[1], -1)


class BottleLinear(Bottle, nn.Linear):
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    pass


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = (
            ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out))

        return ln_out


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, attn_droput=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scaler = np.sqrt(d_model)
        self.dropout = nn.Dropout(attn_droput)
        self.softmax = BottleSoftmax()

    def forward(self, q, k, v, attn_mask=None):
        assert q.size()[-1] == k.size()[-1], ('Dimension -1 of `q` mismatches'
                                              'with dimension -1 of `k`'
                                              ', ({} != {})'.format(
                                                  q.size()[-1], k.size()[-1]))

        assert np.allclose(
            np.sqrt(q.size()[-1]),
            self.scaler), ('Scale size mismatches with dimension -1 of  `q`')

        attn = torch.bmm(q, k.transpose(1, 2)) / self.scaler

        if attn_mask is not None:
            assert attn.size() == attn_mask.size(), (
                'Attention mask shape',
                ' mismatches attention', '  logit shape ({} != {})'.format(
                    attn_mask.size(), attn.size()))
            attn.data.masked_fill_(attn_mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def positional_encode(n_pos, embedded_size):
    data = np.asarray([[
        p / np.power(10000, 2 * i / embedded_size) for i in range(embedded_size)
    ] if p != 0 else np.zeros(embedded_size) for p in range(n_pos)])

    data[1:, 0::2] = np.sin(data[1:, 0::2])
    data[1:, 1::2] = np.cos(data[1:, 1::2])

    return data


class PositionalEncoding(nn.Embedding):

    def __init__(self, n_pos, embedded_size, *args, **kwargs):
        super(PositionalEncoding, self).__init__(n_pos, embedded_size, *args,
                                                 **kwargs)

        encoded = positional_encode(n_pos, embedded_size)
        self.weight.data = torch.FloatTensor(encoded)
