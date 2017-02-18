#!/usr/bin/env python
# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, bprop_len, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        self.iteration = 0
        self.bprop_len = bprop_len

    def create_batch(self):
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
        return [cur_words, next_words]

    def __next__(self):
        #TODO: I really wanna fix this.
        batch = [self.create_batch() for i in range(self.bprop_len)]
        xs = np.array_split(np.array([x for x, _ in batch]),len(batch[0][0]), axis=1)
        xs = [np.squeeze(x) for x in xs]
        ts = np.array_split(np.array([t for _, t in batch]),len(batch[0][0]), axis=1)
        ts = [np.squeeze(x) for x in ts]
        batch = list(zip(xs, ts))
        return batch

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
