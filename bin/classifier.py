# -*- coding: utf-8 -*-
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class RecNetClassifier(link.Chain):
    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(RecNetClassifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.loss = None
        self.accuracy = None

    def __call__(self, *args):
        assert len(args) % 2 == 0
        sep = len(args) // 2
        xs = args[:sep]
        ts = args[sep:]

        self.loss = 0
        ys = self.predictor(*xs)
        for y, t in zip(ys, ts):
            loss = self.lossfun(y, t)
            reporter.report({'loss': loss}, self)
            self.loss += loss
            if self.compute_accuracy:
                self.accuracy = self.accfun(y, t)
                reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
