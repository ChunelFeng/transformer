#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np

import torch
from torch.autograd import Variable

from transformer.modules import (BottleLinear, BottleSoftmax,
                                 PositionalEncoding, ScaledDotProductAttention)


class TestBottleLinear(unittest.TestCase):

    def test_size(self):
        proj = BottleLinear(20, 30)

        x = Variable(torch.FloatTensor(10, 20))
        o = proj(x)
        self.assertEqual(len(o.size()), 2)

        x = Variable(torch.FloatTensor(4, 5, 20))
        o = proj(x)
        self.assertEqual(len(o.size()), len(x.size()))


class TestBottleSoftmax(unittest.TestCase):

    def test_size(self):
        proj = BottleSoftmax()

        x = Variable(torch.FloatTensor(10, 20))
        o = proj(x)
        self.assertEqual(len(o.size()), 2)

        x = Variable(torch.FloatTensor(4, 5, 20))
        o = proj(x)
        self.assertEqual(len(o.size()), len(x.size()))


class TestScaledDotProductAttention(unittest.TestCase):

    def test_size(self):
        q = Variable(torch.FloatTensor(64, 10, 20))
        k = Variable(torch.FloatTensor(64, 10, 21))
        v = Variable(torch.FloatTensor(64, 10, 30))
        with self.assertRaises(AssertionError):
            attention = ScaledDotProductAttention(100)
            attention(q, k, v)

        q = Variable(torch.FloatTensor(64, 10, 20))
        k = Variable(torch.FloatTensor(64, 10, 20))
        v = Variable(torch.FloatTensor(64, 10, 30))
        with self.assertRaises(AssertionError):
            attention = ScaledDotProductAttention(100)
            attention(q, k, v)

        q = Variable(torch.FloatTensor(64, 10, 20))
        k = Variable(torch.FloatTensor(64, 10, 20))
        v = Variable(torch.FloatTensor(64, 10, 30))
        attn_mask = Variable(torch.FloatTensor(64, 10))
        with self.assertRaises(AssertionError):
            attention = ScaledDotProductAttention(100)
            attention(q, k, v, attn_mask)

        q = Variable(torch.FloatTensor(64, 10, 20))
        k = Variable(torch.FloatTensor(64, 10, 20))
        v = Variable(torch.FloatTensor(64, 10, 30))
        attention = ScaledDotProductAttention(20)
        o, attn = attention(q, k, v)
        self.assertEqual(o.size(), v.size())
        self.assertEqual(attn.size(), torch.Size([64, 10, 10]))


class TestPositionalEncoding(unittest.TestCase):

    def test_size(self):
        enc = PositionalEncoding(100, 64)
        self.assertEqual(enc.weight.size(), torch.Size([100, 64]))

    def test_value(self):

        def fn(p, i, d):
            e = p / 10000**(2 * i / d)
            return np.sin(e) if i % 2 == 0 else np.cos(e)

        enc = PositionalEncoding(100, 64)
        self.assertEqual((enc.weight[0] != 0).sum().data[0], 0)
        self.assertTrue(np.allclose(fn(2, 0, 64), enc.weight[2, 0].data[0]))
        self.assertTrue(np.allclose(fn(2, 1, 64), enc.weight[2, 1].data[0]))
        self.assertTrue(np.allclose(fn(2, 2, 64), enc.weight[2, 2].data[0]))
