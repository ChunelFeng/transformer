#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable


class DataLoader(object):

    def __init__(self, cls, *args, **kwargs):
        self.cuda = kwargs.pop('cuda', False)
        self.loader = cls(*args, **kwargs)

    def iter(self, *args, **kwargs):
        iter_ = self.loader.iter(*args, **kwargs)

        for batch in iter_:
            batch = tuple(Variable(torch.LongTensor(obj)) for obj in batch)

            if self.cuda:
                batch = [obj.cuda() for obj in batch]

            yield batch
