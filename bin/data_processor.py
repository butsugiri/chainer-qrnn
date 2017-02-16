# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pickle
from collections import defaultdict
from itertools import groupby, islice


class DataProcessor(object):

    def __init__(self, data_path, test_run):
        self.train_data_path = os.path.join(data_path, "train.txt")
        self.dev_data_path = os.path.join(data_path, "train.txt")
        self.test_data_path = os.path.join(data_path, "train.txt")
        self.test_run = test_run # if true, use tiny datasets for quick test

        self.vocab = defaultdict(lambda: len(self.vocab))
        self.pad_value = self.vocab["<-PAD->"]
        self.vocab["<-UNK->"]

    def prepare_dataset(self):
        # load train/dev/test data
        print("loading dataset...", end='', flush=True, file=sys.stderr)
        if self.test_run:
            print("...preparing tiny dataset for quick test...", end='', flush=True, file=sys.stderr)
        self.train_data = self.load_dataset("train")
        self.dev_data = self.load_dataset("dev")
        self.test_data = self.load_dataset("test")
        print("done", flush=True, file=sys.stderr)

    def load_dataset(self, _type):
        if _type == "train":
            path = self.train_data_path
        elif _type == "dev":
            path = self.dev_data_path
        elif _type == "test":
            path = self.test_data_path

        dataset = []
        question_ids = []
        end = 100 if self.test_run else None
        with open(path, "r") as input_data:
            for line in islice(input_data, end):
                # creating auto encoder for now
                tokens = line.strip().split()
                xs = np.array([self.vocab[t] for t in tokens], dtype=np.int32)
                ys = np.array([self.vocab[t] for t in tokens], dtype=np.int32)
                xs_len = np.array([len(xs)], dtype=np.int32)
                dataset.append((xs, ys, xs_len))
        return dataset
