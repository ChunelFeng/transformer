#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer.layers import EncoderLayer, DecoderLayer


class TestEncoderLayer(unittest.TestCase):

    def test_size(self):
        d_hidden = 20
        n_words = 100
        d_model = 32
        d_emb = d_model
        d_k = 4
        d_v = 5
        n_heads = 8
        batch_size = 32
        max_len = 15
        enc = EncoderLayer(d_model, d_hidden, n_heads, d_k, d_v)
        emb = nn.Embedding(n_words, d_emb)
        x = Variable(
            torch.LongTensor(
                np.random.randint(0, n_words, (batch_size, max_len))))
        embedded = emb(x)

        output, attn = enc(embedded)
        self.assertEqual(output.size(),
                         torch.Size([batch_size, max_len, d_model]))


class TestDecoderLayer(unittest.TestCase):

    def test_size(self):
        d_hidden = 20
        n_words = 100
        d_model = 32
        d_emb = d_model
        d_k = 4
        d_v = 5
        n_heads = 8
        batch_size = 32
        max_len = 15
        enc = EncoderLayer(d_model, d_hidden, n_heads, d_k, d_v)
        enc_emb = nn.Embedding(n_words, d_emb)
        dec_input = Variable(
            torch.LongTensor(
                np.random.randint(0, n_words, (batch_size, max_len))))
        enc_embedded = enc_emb(dec_input)

        enc_output, enc_attn = enc(enc_embedded)
        self.assertEqual(enc_output.size(),
                         torch.Size([batch_size, max_len, d_model]))

        dec = DecoderLayer(d_model, d_hidden, n_heads, d_k, d_v)

        dec_emb = nn.Embedding(n_words, d_emb)
        dec_input = Variable(
            torch.LongTensor(
                np.random.randint(0, n_words, (batch_size, max_len))))
        dec_embedded = dec_emb(dec_input)

        dec_output, dec_slf_attn, inter_attn = dec(dec_embedded, enc_output)
        self.assertEqual(dec_output.size(),
                         torch.Size([batch_size, max_len, d_model]))
