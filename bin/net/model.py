    # -*- coding: utf-8 -*-
import sys
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda, Function, Variable, reporter
from chainer import Link, Chain
from chainer import reporter

class QRNN(Chain):
    def __init__(self, n_vocab, embed_dim, pad_idx, out_size=10, conv_width=2):
        self.embed_dim = embed_dim
        if conv_width ==2:
            super(QRNN, self).__init__(
                embed = L.EmbedID(in_size=n_vocab, out_size=embed_dim, ignore_label=pad_idx),
                W = L.Linear(in_size=embed_dim, out_size=3*out_size, nobias=True),
                V = L.Linear(in_size=embed_dim, out_size=3*out_size),
                l1 = L.Linear(in_size=out_size, out_size=n_vocab)
            )
        else:
            print("未実装")
            raise NotImplementedError

    def __call__(self, *args):
        sep = len(args) // 3
        xs = args[:sep]
        ts = args[sep:sep*2]
        x_len = args[sep*2:]

        inds = self.xp.argsort([-len(x.data) for x in xs]).astype('i')
        xs = [xs[i] for i in inds]
        ts = [ts[i] for i in inds]
        x_len = [int(x_len[i].data) for i in inds]

        emx = [self.embed(x) for x in xs]
        padding = Variable(self.xp.zeros((1, self.embed_dim), dtype=self.xp.float32))
        emx_prev = [F.concat([padding, x[:-1,:]], axis=0) for x in emx]

        emx_prev = F.concat(emx_prev, axis=0)
        emx = F.concat(emx, axis=0)
        split_inds = [sum(x_len[:i]) + x for i, x in enumerate(x_len)][:-1]

        ret = self.W(emx_prev) + self.V(emx)
        pooling_input = F.transpose_sequence(F.split_axis(ret, split_inds, axis=0))

        hs = self.pooling(pooling_input)
        ys = [self.l1(h) for h in hs]
        ts = F.transpose_sequence(ts)

        loss = 0.0
        for y, t in zip(ys, ts):
            loss = F.softmax_cross_entropy(x=y, t=t)
        reporter.report({'loss': loss}, self)
        return loss

    def pooling(self, seqs):
        c_prev = None
        hs = []
        for seq in seqs:
            w0, w1, w2 = F.split_axis(seq, 3, axis=1)
            z = F.tanh(w0)
            f = F.sigmoid(w1)
            o = F.sigmoid(w1)
            if c_prev is None:
                c = (1 - f) * z
            else:
                c_prev = c_prev[:z.shape[0],:]
                c = f * c_prev * (1 - f) * z
            h = o * c
            hs.append(h)
            c_prev = c
        return hs
