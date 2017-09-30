#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from transformer.models import Decoder, Encoder, Transformer


class TestEncoder(unittest.TestCase):

    def test_forward(self):
        batch_size = 2
        vocab_size = 100
        max_len = 5
        n_heads = 8
        d_model = 32
        d_emb = 32
        d_k = 4
        d_v = 4
        n_layers = 6
        d_hidden = 1024
        dropout = 0.1

        x = Variable(torch.LongTensor([[9, 8, 7, 0, 0], [6, 5, 0, 0, 0]]))
        pos = Variable(torch.LongTensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]]))

        enc = Encoder(
            vocab_size,
            max_len,
            n_layers=n_layers,
            d_model=d_model,
            d_emb=d_emb,
            d_hidden=d_hidden,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            pad_id=0)
        enc_outputs, attns = enc(x, pos)
        self.assertEqual(len(enc_outputs), n_layers)
        self.assertTrue(
            np.all([
                x.size() == torch.Size([batch_size, max_len, d_model])
                for x in enc_outputs
            ]))
        self.assertTrue(
            np.all([
                x.size() == torch.Size([n_heads, batch_size, max_len, max_len])
                for x in attns
            ]))


class TestDecoder(unittest.TestCase):

    def test_forward(self):
        batch_size = 2
        vocab_size = 100
        max_len = 5
        n_heads = 8
        d_model = 32
        d_emb = 32
        d_k = 4
        d_v = 4
        n_layers = 6
        d_hidden = 1024
        dropout = 0.1

        tgt_seq = Variable(torch.LongTensor([[9, 8, 7, 0, 0], [6, 5, 0, 0, 0]]))
        tgt_pos = Variable(torch.LongTensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]]))

        src_max_len = 4
        src_seq = Variable(torch.LongTensor([[9, 8, 7, 0], [6, 5, 0, 0]]))
        enc_output = Variable(
            torch.FloatTensor(
                np.random.randn(batch_size, src_max_len, d_model)))

        dec = Decoder(
            vocab_size,
            max_len,
            n_layers=n_layers,
            d_model=d_model,
            d_emb=d_emb,
            d_hidden=d_hidden,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            pad_id=0)
        dec_outputs, slf_attns, inter_attns = dec(tgt_seq, tgt_pos, src_seq,
                                                  enc_output)
        self.assertEqual(len(dec_outputs), n_layers)
        self.assertTrue(
            np.all([
                x.size() == torch.Size([batch_size, max_len, d_model])
                for x in dec_outputs
            ]))
        self.assertTrue(
            np.all([
                x.size() == torch.Size([n_heads, batch_size, max_len, max_len])
                for x in slf_attns
            ]))
        self.assertTrue(
            np.all([
                x.size() == torch.Size([
                    n_heads, batch_size, max_len, src_max_len
                ]) for x in inter_attns
            ]))
