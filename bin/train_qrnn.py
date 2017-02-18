# -*- coding: utf-8 -*-
import argparse
import numpy as np

import chainer
from chainer import reporter, training, cuda
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer.training import extensions

from net.model import QRNNLangModel
from utils import convert
from data_processor import DataProcessor
from classifier import RecNetClassifier
from iterator import ParallelSequentialIterator
from updater import BPTTUpdater


def main(args):
    # load data
    dp = DataProcessor(args.data, args.test)
    dp.prepare_dataset()
    train_data = dp.train_data
    dev_data = dp.dev_data
    test_data = dp.test_data

    # create model
    vocab = dp.vocab
    embed_dim = args.dim
    out_size = args.out_size
    model = RecNetClassifier(QRNNLangModel(n_vocab=len(vocab), embed_dim=embed_dim, out_size=out_size))

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = O.AdaGrad(lr=args.lr)
    optimizer.setup(model)

    # create iterators from loaded data
    train_iter = ParallelSequentialIterator(train_data, args.batchsize, bprop_len=30)
    dev_iter = ParallelSequentialIterator(dev_data, args.batchsize, repeat=False, bprop_len=30)

    updater = BPTTUpdater(train_iter, optimizer, device=args.gpu, converter=convert, bprop_len=30)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    # setup evaluation
    eval_model = model.copy()
    eval_model.train = False
    trainer.extend(extensions.Evaluator(
        dev_iter, eval_model, device=args.gpu, converter=convert))

    # extentions...
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'val_perplexity', 'perplexity']))
    trainer.extend(extensions.ProgressBar(update_interval=5))
    # take a shapshot when the model achieves highest accuracy in dev set
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}',
        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss')))

    # trainer.extend(extensions.ExponentialShift("lr", 0.5, optimizer=optimizer),
    #                trigger=chainer.training.triggers.MaxValueTrigger("validation/main/map"))
    trainer.run()

def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,
                        default=-1, help='GPU ID (Negative value indicates CPU)')
    parser.add_argument('--epoch', dest='epoch', type=int,
                        default=100, help='Number of times to iterate through the dataset')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=3, help='Minibatch size')
    parser.add_argument('--data',  type=str,
                        default='../data/ptb', help='Path to input (train/dev/test) data file')
    parser.add_argument('--dim',  type=int,
                        default=10, help='embed dimension')
    parser.add_argument('--glove', action='store_true',
                        help='Use GloVe vector?')
    parser.set_defaults(glove=False)
    parser.add_argument('--glove-path', dest='glove_path', type=str,
                        default="../../disco_parse/data/glove_model/glove.6B.100d.txt", help='Path to pretrained glove vector')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny dataset for quick test')
    parser.set_defaults(test=False)
    parser.add_argument('--decay',  type=float,
                        default=0.0004, help='Weight decay rate')
    parser.add_argument('--lr', type=float,
                        default=0.04, help='Learning Rate')
    parser.add_argument('--unit', type=int,
                        default=10, help='Hidden dim')
    args = parser.parse_args()

    main(args)
