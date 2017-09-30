#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import unittest

import transformer.utils as utils


class TestUtils(unittest.TestCase):

    def test_padding_mask(self):
        q = torch.LongTensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        k = torch.LongTensor([[2, 3, 0, 0], [1, 4, 5, 0]])
        expected = torch.ByteTensor([[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1],
                                      [0, 0, 1, 1], [0, 0, 1, 1]],
                                     [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
                                      [0, 0, 0, 1], [0, 0, 0, 1]]])
        mask = utils.padding_mask(q, k)
        self.assertEqual(mask.size(), torch.Size([2, 5, 4]))
        self.assertTrue(torch.equal(mask, expected))

    def test_subsequent_mask(self):
        seqs = torch.LongTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        mask = utils.subsequent_mask(seqs)
        mask = mask.numpy().tolist()

        for k, seq in enumerate(mask):
            for i, row in enumerate(seq):
                for j, m in enumerate(row):
                    if j <= i:
                        self.assertTrue(m == 0)
                    else:
                        self.assertTrue(m == 1)
