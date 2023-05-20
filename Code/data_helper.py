import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from sklearn.model_selection import train_test_split
import json
import time
from torchtext import data, datasets, vocab
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, math
from numpy.random import seed
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import random, tqdm, sys, math, gzip
import nlp


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        assert len(inputs) == len(
            targets
        ), "Length of inputs and targets should be same."
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def create_datasets(arg):
    if arg.task == "TOTTO":
        return create_totto_datasets(arg)
    elif arg.task == "cnn_dailymail":
        return create_cnn_dailymail_datasets(arg)
    else:
        assert False, f"Data fetching for {arg.task} not defined"


# For Totto, we do not have the reference sentences for test set.
# So, we can divide the training set into train and validation
# and use actual validation set as test set
def read_file(arg, f):
    with open(f) as fp:
        lines = fp.readlines()
        inputs = []
        targets = []
        for line in lines:
            # load json data on each line
            entry = json.loads(line)
            # print(entry)
            if arg.input_string == "raw_table":
                target = entry["sentence_annotations"][0]["final_sentence"]
                entry.pop("sentence_annotations")
                input = json.dumps(entry)
            elif arg.input_string == "subtable_str_plus_subtable_metadata_str":
                target = entry["sentence_annotations"][0]["final_sentence"]
                input = entry["subtable_metadata_str"] + " " + entry["subtable_str"]
            inputs.append(input)
            targets.append(target)
    return inputs, targets


def create_totto_datasets(arg):
    print("Creating Totto datasets..")
    # arg.train_input, arg.development_input
    start = time.time()
    if arg.toy_dataset:
        print("Using toy dataset..")

    inputs, targets = read_file(arg, arg.train_input)
    if arg.toy_dataset:
        inputs = inputs[: arg.toy_dataset]
        targets = targets[: arg.toy_dataset]

    num_datapoints = len(inputs)
    val_set_size = max(1, int(0.125 * num_datapoints))
    train, val = train_test_split(
        inputs, test_size=val_set_size, shuffle=False, random_state=0
    )
    train_t, val_t = train_test_split(
        targets, test_size=val_set_size, shuffle=False, random_state=0
    )

    training_set = Dataset(train, train_t)
    validation_set = Dataset(val, val_t)
    assert len(validation_set) == val_set_size, "Validation size not matching"
    assert len(training_set) == (
        num_datapoints - val_set_size
    ), "Training size not matching"

    inputs, targets = read_file(arg, arg.development_input)
    if arg.toy_dataset:
        inputs = inputs[: arg.toy_dataset]
        targets = targets[: arg.toy_dataset]
    test_set = Dataset(inputs, targets)
    end = time.time()
    print("Time taken to create datasets is %0.2f mins" % ((end - start) / 60))
    return training_set, validation_set, test_set


def create_cnn_dailymail_datasets(arg):
    def convert(dataset, arg):
        inputs = []
        targets = []
        for i in np.arange(len(dataset)):
            i = int(i)
            # print(dataset[i])
            inputs.append(dataset[i]["article"])
            targets.append(dataset[i]["highlights"])
        if arg.toy_dataset:
            inputs = inputs[: arg.toy_dataset]
            targets = targets[: arg.toy_dataset]
        return Dataset(inputs, targets)

    train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
    val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:1%]")
    test_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="test[:1%]")

    training_set = convert(train_dataset, arg)
    validation_set = convert(val_dataset, arg)
    test_set = convert(test_dataset, arg)
    return training_set, validation_set, test_set


def create_dataloaders(arg, training_set, validation_set, test_set):
    start = time.time()
    trainloader, valloader, testloader = None, None, None
    if training_set is not None:
        trainloader = torch.utils.data.DataLoader(
            training_set,
            batch_size=arg.batch_size,
            shuffle=arg.shuffle_train,
            num_workers=2,
        )
    if validation_set is not None:
        valloader = torch.utils.data.DataLoader(
            validation_set, batch_size=arg.batch_size, shuffle=False, num_workers=2
        )
    if test_set is not None:
        testloader = torch.utils.data.DataLoader(
            test_set, batch_size=arg.batch_size, shuffle=False, num_workers=2
        )
    end = time.time()
    return trainloader, valloader, testloader
