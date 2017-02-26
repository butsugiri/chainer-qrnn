# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
from datetime import datetime
import json
import random

import chainer
from chainer import reporter, training, cuda
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer.training import extensions

from QRNNLM.net.model import QRNNLangModel
from QRNNLM.utils import convert, ThresholdTrigger
from QRNNLM.data_processor import DataProcessor
from QRNNLM.classifier import RecNetClassifier
from QRNNLM.iterator import ParallelSequentialIterator
from QRNNLM.updater import BPTTUpdater

def set_random_seed(seed):
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)

def main(args):
    # use same seed
    set_random_seed(0)

    # create output dir
    start_time = datetime.now().strftime('%m%d_%H_%M_%S')
    dest = os.path.join("../result/", start_time)
    os.makedirs(dest)
    abs_dest = os.path.abspath(dest)
    with open(os.path.join(dest, "settings.json"), "w") as fo:
        print("Log files are saved in {}".format(abs_dest, file=sys.stderr))
        fo.write(json.dumps(vars(args), sort_keys=True, indent=4))
        print(json.dumps(vars(args), sort_keys=True, indent=4), file=sys.stderr)

    # load data
    dp = DataProcessor(args.data, args.test)
    dp.prepare_dataset()
    train_data = dp.train_data
    dev_data = dp.dev_data
    test_data = dp.test_data

    # create model
    vocab = dp.vocab
    embed_dim = args.dim
    out_size = args.unit
    model = RecNetClassifier(QRNNLangModel(n_vocab=len(vocab), embed_dim=embed_dim, out_size=out_size))

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = O.SGD(lr=args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.clipping))
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.decay))

    # create iterators from loaded data
    bprop_len = args.bproplen
    train_iter = ParallelSequentialIterator(train_data, args.batchsize, bprop_len=bprop_len)
    dev_iter = ParallelSequentialIterator(dev_data, 1, repeat=False, bprop_len=bprop_len)

    updater = BPTTUpdater(train_iter, optimizer, device=args.gpu, converter=convert)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=abs_dest)

    # setup evaluation
    eval_model = model.copy()
    eval_model.predictor.train = False
    trainer.extend(extensions.Evaluator(
        dev_iter, eval_model, device=args.gpu, converter=convert, eval_hook=lambda _: eval_model.predictor.reset_state()))

    # extentions...
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'val_perplexity', 'perplexity']))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # take a shapshot when the model achieves highest accuracy in dev set
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}',
        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss')))

    trainer.extend(extensions.ExponentialShift("lr", 0.95, optimizer=optimizer),
                   trigger=ThresholdTrigger(1, 'epoch', 6))
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
                        default=72, help='Number of times to iterate through the dataset')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=20, help='Minibatch size')
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
                        default=0.0002, help='Weight decay rate')
    parser.add_argument('--lr', type=float,
                        default=1.0, help='Learning Rate')
    parser.add_argument('--unit', type=int,
                        default=10, help='Hidden dim')
    parser.add_argument('--bproplen', type=int,
                        default=105, help='Backprop Length for BPTT')
    parser.add_argument('--clipping', type=int,
                        default=10, help='GradientClipping Threshold')
    args = parser.parse_args()

    main(args)
