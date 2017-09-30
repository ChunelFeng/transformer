#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import torch

from dataloader import DataLoader
from poketto.nlp.data import MachineTranslationDataLoader
from poketto.pytorch.train import Trainer
from transformer.models import Transformer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-src', type=str, required=True)
    parser.add_argument('-tgt', type=str, required=True)

    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-max_vocab_size', type=int, default=90000)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-max_len', type=int, default=50)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_hidden', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-logdir', type=str, default='/tmp/transformer')
    parser.add_argument('-save_model', default=None)
    parser.add_argument(
        '-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-cuda', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    loader = DataLoader(
        MachineTranslationDataLoader,
        args.src,
        args.tgt,
        max_vocab_size=args.max_vocab_size,
        min_word_count=args.min_word_count,
        max_len=args.max_len,
        cuda=args.cuda)

    src_vocab, tgt_vocab = loader.loader.src.vocab, loader.loader.tgt_in.vocab
    print(len(src_vocab), len(tgt_vocab))

    torch.save(src_vocab, os.path.join(args.logdir, 'src_vocab.pt'))
    torch.save(tgt_vocab, os.path.join(args.logdir, 'tgt_vocab.pt'))

    transformer = Transformer(
        len(src_vocab),
        len(tgt_vocab),
        args.max_len + 2,
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_emb=args.d_model,
        d_hidden=args.d_hidden,
        n_heads=args.n_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=args.dropout,
        pad_id=src_vocab.pad_id)

    weights = torch.ones(len(tgt_vocab))
    weights[tgt_vocab.pad_id] = 0

    optimizer = torch.optim.Adam(
        transformer.get_trainable_parameters(), lr=args.lr)

    loss_fn = torch.nn.CrossEntropyLoss(weights)

    if args.cuda:
        transformer = transformer.cuda()
        loss_fn = loss_fn.cuda()

    def loss_fn_wrap(src, tgt_in, tgt_out, src_pos, tgt_pos, logits):
        return loss_fn(logits, tgt_out.contiguous().view(-1))

    def get_performance(gold, logits, pad_id):
        gold = gold.contiguous().view(-1)
        logits = logits.max(dim=1)[1]

        n_corrects = logits.data.eq(gold.data)
        n_corrects = n_corrects.masked_select(gold.ne(pad_id).data).sum()

        return n_corrects

    def epoch_fn(epoch, stats):
        (n_corrects, n_words
         ) = list(zip(* [(x['n_corrects'], x['n_words']) for x in stats]))

        train_acc = sum(n_corrects) / sum(n_words)

        return {'train_acc': train_acc}

    def step_fn(step, src, tgt_in, tgt_out, src_pos, tgt_pos, logits):
        n_corrects = get_performance(tgt_out, logits, tgt_vocab.pad_id)
        n_words = tgt_out.data.ne(tgt_vocab.pad_id).sum()

        return {'n_corrects': n_corrects, 'n_words': n_words}

    trainer = Trainer(
        transformer,
        loss_fn_wrap,
        optimizer,
        logdir=args.logdir,
        hparams=args,
        save_mode=args.save_mode)

    trainer.train(
        lambda: loader.iter(batch_size=args.batch_size, with_pos=True),
        epochs=args.epochs,
        epoch_fn=epoch_fn,
        step_fn=step_fn,
        metric='train_acc')


if __name__ == '__main__':
    main()
