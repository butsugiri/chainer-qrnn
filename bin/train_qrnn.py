# coding: utf-8
import os
import json
import sys
import argparse
from datetime import datetime
import numpy as np

import chainer
from chainer import reporter, training, cuda
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer.functions import sigmoid_cross_entropy, binary_accuracy
from chainer.training import extensions

from net.model import QRNN
from utils import convert
from data_processor import DataProcessor
from chainer.dataset import concat_examples
from functools import partial




def main(args):
    # load data
    dp = DataProcessor(args.data, args.test)
    dp.prepare_dataset()
    train_data = dp.train_data
    dev_data = dp.dev_data
    test_data = dp.test_data

    # create model
    vocab = dp.vocab
    pad_idx = dp.pad_value
    embed_dim = args.dim
    model = QRNN(n_vocab=len(vocab), embed_dim=embed_dim, pad_idx=pad_idx)  # ABCNNはoutput = 50固定らしいが．

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = O.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize, shuffle=True)

    dev_train_iter = chainer.iterators.SerialIterator(
        train_data, args.batchsize, repeat=False)
    if args.use_test_data:
        dev_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False)
    else:
        dev_iter = chainer.iterators.SerialIterator(dev_data, args.batchsize, repeat=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=convert)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    # setup evaluation
    # eval_predictor = model.copy().predictor
    # eval_predictor.train = False
    # trainer.extend(extensions.Evaluator(
    #     dev_iter, eval_predictor, device=args.gpu))

    # extentions...
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    # trainer.extend(extensions.ProgressBar(update_interval=10))
    # take a shapshot when the model achieves highest accuracy in dev set
    # trainer.extend(extensions.snapshot_object(
    #     model, 'model_epoch_{.updater.epoch}',
    #     trigger=chainer.training.triggers.MaxValueTrigger('validation/main/map')))

    # trainer.extend(extensions.ExponentialShift("lr", 0.5, optimizer=optimizer),
    #                trigger=chainer.training.triggers.MaxValueTrigger("validation/main/map"))
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,
                        default=-1, help='GPU ID (Negative value indicates CPU)')
    parser.add_argument('--epoch', dest='epoch', type=int,
                        default=5, help='Number of times to iterate through the dataset')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=3, help='Minibatch size')
    parser.add_argument('--data',  type=str,
                        default='../data/', help='Path to input (train/dev/test) data file')
    parser.add_argument('--dim',  type=int,
                        default=10, help='embed dimension')
    parser.add_argument('--glove', action='store_true',
                        help='Use GloVe vector?')
    parser.set_defaults(glove=False)
    parser.add_argument('--glove-path', dest='glove_path', type=str,
                        default="../../disco_parse/data/glove_model/glove.6B.100d.txt", help='Path to pretrained glove vector')
    parser.add_argument('--word2vec', action='store_true',
                        help='Use word2vec vector?')
    parser.set_defaults(word2vec=False)
    parser.add_argument('--word2vec-path', dest='word2vec_path', type=str,
                        default="../../disco_parse/data/glove_model/glove.6B.100d.txt", help='Path to pretrained word2vec vector')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny dataset for quick test')
    parser.set_defaults(test=False)
    parser.add_argument('--decay',  type=float,
                        default=0.0002, help='Weight decay rate')
    parser.add_argument('--vocab',  type=str,
                        default="../work/word2vec_vocab.txt", help='Vocabulary file')
    parser.add_argument('--lr',  type=float,
                        default=0.08, help='Learning rate')
    parser.add_argument('--use-test-data', dest="use_test_data", action='store_true',
                        help='Use test data instead of dev data')
    parser.set_defaults(use_test_data=False)
    args = parser.parse_args()

    main(args)
