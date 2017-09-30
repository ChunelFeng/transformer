#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer.sublayers import PositionwiseFeedForward, MultiHeadAttention


class TestPositionwiseFeedForward(unittest.TestCase):

    def test_size(self):
        emb = nn.Embedding(10, 4)
        x = Variable(torch.LongTensor([[1, 4, 5, 0, 0], [2, 3, 1, 5, 0]]))
        embedded = emb(x)

        fc = PositionwiseFeedForward(10, 4)
        o = fc(embedded)

        self.assertEqual(o.size(), torch.Size([2, 5, 4]))


class TestMultiHeadAttention(unittest.TestCase):

    def test_mistmatch_proj_subspace_dimension(self):
        with self.assertRaises(AssertionError):
            n_heads = 8
            d_k = 10
            d_v = 20
            d_model = n_heads * d_k + 1
            MultiHeadAttention(
                n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model)

    def test_size(self):
        batch_size = 32
        max_len = 15
        emb = nn.Embedding(10, 32)
        x = Variable(
            torch.LongTensor(
                np.random.randint(0, 10, batch_size * max_len).reshape(
                    batch_size, max_len)))
        embedded = emb(x)

        n_heads = 8
        d_k = 4
        d_model = 32
        attn = MultiHeadAttention(
            n_heads=n_heads, d_k=d_k, d_v=d_k, d_model=d_model)
        o, attns = attn(embedded, embedded, embedded)

        self.assertEqual(o.size(), torch.Size([batch_size, max_len, d_model]))
        self.assertEqual(attns.size(),
                         torch.Size([n_heads, batch_size, max_len, max_len]))
