#!/usr/bin/env python
# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Variable

import numpy as np


class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device, converter):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.converter = converter

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()

        in_tuples = self.converter(batch, self.device)
        in_vars = [Variable(x) for x in in_tuples]

        loss += optimizer.target(*in_vars)

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters
