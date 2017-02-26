# -*- coding: utf-8 -*-
import sys
import os
import argparse
import json

import numpy as np
import chainer
from chainer.training import extensions
import chainer.serializers as S

from QRNNLM.net.model import QRNNLangModel
from QRNNLM.utils import convert
from QRNNLM.data_processor import DataProcessor
from QRNNLM.classifier import RecNetClassifier
from QRNNLM.iterator import ParallelSequentialIterator

def main(args):
    # load config file and obtain embed dimension and hidden dimension
    with open(args.config_path, 'r') as fi:
        config = json.load(fi)
        embed_dim = config["dim"]
        hidden_dim = config["unit"]
        print("Embedding Dimension: {}\nHidden Dimension: {}\n".format(embed_dim, hidden_dim), file=sys.stderr)

    # load data
    dp = DataProcessor(data_path=config["data"], test_run=False)
    dp.prepare_dataset()

    # create model
    vocab = dp.vocab
    model = RecNetClassifier(QRNNLangModel(n_vocab=len(vocab), embed_dim=embed_dim, out_size=hidden_dim))

    # load parameters
    print("loading paramters to model...", end='', file=sys.stderr, flush=True)
    S.load_npz(filename=args.model_path, obj=model)
    print("done.", file=sys.stderr, flush=True)

    # create iterators from loaded data
    bprop_len = config["bproplen"]
    test_data = dp.test_data
    test_iter = ParallelSequentialIterator(test_data, 1, repeat=False, bprop_len=bprop_len)

    # evaluate the model
    print('testing...', end='', file=sys.stderr, flush=True)
    model.predictor.reset_state()
    model.predictor.train = False
    evaluator = extensions.Evaluator(test_iter, model, converter=convert)
    result = evaluator()
    print('done.\n', file=sys.stderr, flush=True)
    print('Perplexity: {}'.format(np.exp(float(result['main/loss']))), end='', file=sys.stderr, flush=True)

def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])

if __name__ == "__main__":
    """
    settings.json is automatically created by the train_qrnn.py
    """
    parser = argparse.ArgumentParser(description="Evaluation Script for Trained QRNN Model")
    parser.add_argument('--config-path', required=True, dest="config_path", type=str, help='Path to config (settings.json) file')
    parser.add_argument('--model-path', required=True, dest="model_path", type=str, help='Path to model file')
    args = parser.parse_args()
    main(args)
