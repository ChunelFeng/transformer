#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable

from dataloader import DataLoader
from poketto.nlp import Beam
from poketto.nlp.data import TextLineDataLoader
from transformer.models import Transformer

plt.style.use('seaborn')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'].insert(0, 'MS Gothic')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-src', type=str, required=True)

    parser.add_argument('-src_vocab', type=str, required=True)
    parser.add_argument('-tgt_vocab', type=str, required=True)

    parser.add_argument('-chkpt', type=str, required=True)
    parser.add_argument('-logdir', type=str, default='log/')

    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-beam_size', type=int, default=5)

    parser.add_argument('-cuda', action='store_true')

    args = parser.parse_args()

    return args


class Translator(object):

    def __init__(self, src_vocab, tgt_vocab, checkpoint, opts):

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        hparams = checkpoint['hparams']

        transformer = Transformer(
            len(src_vocab),
            len(tgt_vocab),
            hparams.max_len + 2,
            n_layers=hparams.n_layers,
            d_model=hparams.d_model,
            d_emb=hparams.d_model,
            d_hidden=hparams.d_hidden,
            n_heads=hparams.n_heads,
            d_k=hparams.d_k,
            d_v=hparams.d_v,
            dropout=hparams.dropout,
            pad_id=src_vocab.pad_id)

        transformer.load_state_dict(checkpoint['model'])
        log_proj = torch.nn.LogSoftmax()

        if hparams.cuda:
            transformer.cuda()
            log_proj.cuda()

        transformer.eval()

        self.hparams = hparams
        self.opts = opts
        self.model = transformer
        self.log_proj = log_proj

    def translate(self, src_batch):
        src_seq, src_pos = src_batch

        batch_size, max_time = src_seq.size()
        beam_size = self.opts.beam_size

        # Encode
        enc_outputs, _ = self.model.encoder(src_seq, src_pos)
        src_seq = Variable(
            src_seq.data.view(batch_size, -1).repeat(1, beam_size).view(
                -1, max_time))
        enc_outputs = [
            Variable(
                o.data.view(batch_size, -1).repeat(1, beam_size).view(
                    batch_size * beam_size, max_time, -1)) for o in enc_outputs
        ]

        attns = [None for i in range(batch_size)]

        beams = [
            Beam(
                size=beam_size,
                bos_id=self.tgt_vocab.bos_id,
                eos_id=self.tgt_vocab.eos_id) for i in range(batch_size)
        ]

        batch_idx = list(range(batch_size))
        remains = batch_idx
        n_remains = len(remains)

        # Decode for each time step
        for i in range(self.hparams.max_len):
            time_step = i + 1

            # B x M
            beam_input = torch.stack(
                [torch.LongTensor(b.get_hyps()) for b in beams if not b.done()])
            # BM x T (T0 = 1)
            beam_input = beam_input.view(-1, time_step)
            beam_input = Variable(beam_input, volatile=True)

            # T
            beam_pos = torch.arange(1, time_step + 1).long().unsqueeze(dim=0)
            # BM x T
            beam_pos = Variable(
                beam_pos.repeat(beam_size * n_remains, 1), volatile=True)

            if self.opts.cuda:
                beam_input = beam_input.cuda()
                beam_pos = beam_pos.cuda()

            dec_outputs, _, inter_attns = self.model.decoder(
                beam_input, beam_pos, src_seq, enc_outputs)
            dec_output = dec_outputs[-1][:, -1, :]
            # BM x V
            logits = self.model.proj(dec_output)
            logits = self.log_proj(logits)
            # B x M x V
            logits = logits.view(n_remains, beam_size, -1).cpu()

            remains = []
            for i in range(batch_size):
                if beams[i].done():
                    continue

                if not beams[i].advance(logits.data.numpy()[batch_idx[i]]):
                    remains += [i]
                else:
                    inter_attn = inter_attns[-1]
                    n_heads, bmb, len_q, len_k = inter_attn.size()
                    attn_sizes = [
                        n_heads, bmb // beam_size, beam_size, len_q, len_k
                    ]
                    inter_attn = inter_attn.view(attn_sizes)
                    attns[i] = inter_attn[:, batch_idx[i],
                                          0, :, :].data.cpu().numpy()

            if not remains:
                break

            remain_idx = torch.LongTensor([batch_idx[i] for i in remains])
            if self.opts.cuda:
                remain_idx = remain_idx.cuda()

            batch_idx = {beam: i for i, beam in enumerate(remains)}
            n_remains = len(remains)

            src_seq = src_seq.data.view(src_seq.size()[0] // beam_size, -1)
            src_seq = src_seq.index_select(dim=0, index=remain_idx)
            src_seq = src_seq.view(n_remains * beam_size, -1)
            src_seq = Variable(src_seq, volatile=True)

            def filter_enc_output(o):
                o = o.data.view(o.size()[0] // beam_size, -1)
                o = o.index_select(dim=0, index=remain_idx)
                o = o.view(n_remains * beam_size, max_time, -1)
                o = Variable(o, volatile=True)
                return o

            enc_outputs = [filter_enc_output(o) for o in enc_outputs]

        return [b.get_hyps()[0][1:] for b in beams], attns


def visaulize_attention(src_words, tgt_words, inter_attns, save_path):
    for (src, tgt, attns) in zip(src_words, tgt_words, inter_attns):
        name = '{src}_{tgt}'.format(src=''.join(src), tgt=''.join(tgt))

        for i, attn in enumerate(attns):
            fig = plt.figure()
            heatmap = plt.pcolor(attn, cmap=plt.cm.Blues)
            ax = plt.gca()
            plt.colorbar(heatmap)
            ax.set_xticks(np.arange(len(src)) + 0.5, minor=False)
            ax.set_yticks(np.arange(len(tgt)) + 0.5, minor=False)
            ax.set_xticklabels(src, minor=False)
            ax.set_yticklabels(tgt, minor=False)
            plt.savefig(
                os.path.join(save_path, '{name}_{head}.png'.format(
                    name=name, head=i)))
            plt.close(fig)


if __name__ == '__main__':
    args = parse_args()

    src_vocab = torch.load(args.src_vocab)
    tgt_vocab = torch.load(args.tgt_vocab)
    loader = DataLoader(
        TextLineDataLoader, args.src, vocab=src_vocab, cuda=args.cuda)

    checkpoint = torch.load(args.chkpt)
    translator = Translator(src_vocab, tgt_vocab, checkpoint, args)

    all_translations = []
    for batch in loader.iter(args.batch_size, shuffle=False, with_pos=True):
        translations, inter_attns = translator.translate(batch)

        # Ignore <eos>
        src = batch[0].data.cpu().numpy()
        src_len = np.not_equal(src, src_vocab.pad_id).sum(axis=1)
        src = [x[:src_len[i]] for i, x in enumerate(src)]
        translations = [x[:-1] for x in translations]
        inter_attns = [
            attn[:, :-1, :src_len[i]] for i, attn in enumerate(inter_attns)
        ]

        src_words = src_vocab.inverse_transform(src)
        tgt_words = tgt_vocab.inverse_transform(translations)

        all_translations += tgt_words

        visaulize_attention(src_words, tgt_words, inter_attns, args.logdir)

        print('\n'.join([''.join(x) for x in tgt_words]))

    output = os.path.join(
        args.logdir, '{base}.trans.txt'.format(base=os.path.basename(args.src)))

    with open(output, mode='w') as f:
        f.writelines('\n'.join([''.join(x) for x in all_translations]))
