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

    xs = [to_device(x) for x, _, in batch]
    ts = [to_device(t) for _, t, in batch]
    out = tuple(xs + ts)
    return out

class ThresholdTrigger(object):
    def __init__(self, period, unit, threshold):
        self.period = period
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit
        self.count = 0
        self.threshold = threshold

    def __call__(self, trainer):
        updater = trainer.updater
        if self.unit == 'epoch':
            prev = self.count
            self.count = updater.epoch_detail // self.period
            return (prev != self.count) and (self.count > self.threshold)
        else:
            iteration = updater.iteration
            return iteration > 0 and iteration % self.period == 0
