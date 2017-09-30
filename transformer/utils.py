#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def _get_tensor(*args):
    return [t.data if isinstance(t, Variable) else t for t in args]


def padding_mask(q, k, pad_id=0):
    q, k = _get_tensor(q, k)
    batch_size, len_q = q.size()
    batch_size, len_k = k.size()
    mask = torch.eq(k, pad_id).unsqueeze(1).expand(batch_size, len_q, len_k)

    return mask


def subsequent_mask(seq):
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = torch.from_numpy(mask)
    if seq.is_cuda:
        mask = mask.cuda()

    return mask
