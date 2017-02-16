# -*- coding: utf-8 -*-
import sys
import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

def convert(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    xs = [to_device(x) for x, _, _ in batch]
    ts = [to_device(t) for _, t, _ in batch]
    x_len = [to_device(l) for _, _, l in batch]
    out = tuple(xs + ts + x_len)
    return out
