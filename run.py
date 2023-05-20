import os
import numpy as np
import time
import csv
import pandas as pd
import sys
import tqdm
from datetime import date
import argparse
import matplotlib.pyplot as plt
import json
import torch

from Code.data_helper import *
from Code.metrics import *
from Models.Baselines import *

def train(arg):
    enter_at = time.time()
    assert arg.mode == 'train', f'arg.mode must be set to train'

    training_set, validation_set, test_set = create_datasets(arg)
    train_loader, validation_loader, test_loader = create_dataloaders(arg, training_set, validation_set, test_set)  

    print('training examples', len(training_set))
    print('validation examples', len(validation_set))
    print('test examples', len(test_set))  

    if (arg.cuda):
        gpu_available = torch.cuda.is_available()
        assert gpu_available, 'GPU not available'
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    if (arg.model == 'Bert2Bert'):
        model = Bert2Bert()
    else:
        assert False, f'Model {arg.model} is not defined'

    model.to(device)

    optimizer = torch.optim.Adam(lr=arg.lr, 
                                 params=model.parameters())

    best_model_loss = 1e100
    checkpoint = arg.checkpoint_path
    for epoch in np.arange(arg.num_epochs):
        print(f'Epoch {epoch}')

        start = time.time()
        model.train()  
        train_loss = 0.0 
        list_batch_scores = []
        for batch_idx, data in enumerate(train_loader):
            start1 = time.time()
            optimizer.zero_grad()
            inputs, targets = data
            inputs = list(inputs)
            targets = list(targets)
            # print(inputs, targets)

            model_inputs = model.get_model_inputs(inputs, targets, device)
            outputs = model(**model_inputs)

            loss = model.get_loss(outputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # generated_sentences = model.generate(list(inputs), device)
            # avg_batch_score = compute_metric(arg, targets, generated_sentences)
            avg_batch_score = -1
            list_batch_scores.append(avg_batch_score)
            # for t, p in zip(targets, generated_sentences):
            #     print('TARGET=>', t)
            #     print('GENERATED=>', p)
            #     print('---------')

            end1 = time.time()
            print('(train) Epoch %d, batch %d, loss %0.4f, avg_score %0.4f, time %0.3f mins'%
            (epoch, batch_idx, loss.item(), avg_batch_score, ((end1-start1)/60)))
            del outputs
        train_loss = train_loss/len(training_set)
        train_avg_score = np.mean(list_batch_scores)

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            list_batch_scores = []
            for batch_idx, data in enumerate(validation_loader):   
                start1 = time.time()
                inputs, targets = data
                inputs = list(inputs)
                targets = list(targets)
                
                model_inputs = model.get_model_inputs(inputs, targets, device)
                outputs = model(**model_inputs)
                
                loss = model.get_loss(outputs)
                val_loss += loss.item()

                generated_sentences = model.generate(list(inputs), device)
                avg_batch_score = compute_metric(arg, targets, generated_sentences)
                list_batch_scores.append(avg_batch_score)

                end1 = time.time()
                print('(val) Epoch %d, batch %d, loss %0.4f, avg_score %0.4f, time %0.3f mins'%
                (epoch, batch_idx, loss.item(), avg_batch_score, ((end1-start1)/60)))
                del outputs
            val_loss = val_loss/len(validation_set)
            val_avg_score = np.mean(list_batch_scores)

            end = time.time()
            print('--> Epoch %d, train loss %0.4f (%s %0.4f), val loss %0.4f (%s %0.4f), time on train+val set %0.3f mins'%
            (epoch, train_loss, arg.metric, train_avg_score, val_loss, arg.metric, val_avg_score, (end-start)/60))

            if (best_model_loss > val_loss):
                print(f'Got a new best model with loss {val_loss}, previous best was {best_model_loss}.')
                best_model_loss = val_loss
                torch.save(model, checkpoint)
    
    #TODO: Measure BLEU and PARENT score instead of loss for test and val
    del model
    with torch.no_grad():
        start = time.time()
        model = torch.load(checkpoint)
        model.eval()
        model.to(device)
        test_loss = 0.0
        list_batch_scores = []
        for batch_idx, data in enumerate(test_loader):
            start1 = time.time()   
            inputs, targets = data
            inputs = list(inputs)
            targets = list(targets)

            model_inputs = model.get_model_inputs(inputs, targets, device)
            outputs = model(**model_inputs)

            loss = model.get_loss(outputs)
            test_loss += loss.item()

            generated_sentences = model.generate(list(inputs), device)
            avg_batch_score = compute_metric(arg, targets, generated_sentences)
            list_batch_scores.append(avg_batch_score)

            end1 = time.time()
            print('(test) batch %d, loss %0.4f, avg_score %0.4f, time %0.3f mins'%
            (batch_idx, loss.item(), avg_batch_score, ((end1-start1)/60)))
            del outputs
        test_loss = test_loss/len(test_set)
        test_avg_score = np.mean(list_batch_scores)

        end = time.time()
        print('--> Best Model %0.4f, Best Test loss %0.4f (%s %0.4f), time on test set %0.3f mins'%
        (best_model_loss, test_loss, arg.metric, test_avg_score, (end-start)/60))

    exit_at = time.time()
    print(f'Time taken to train is {(exit_at - enter_at)/60} mins.')
    return

def generate_sentences(arg):
    enter_at = time.time()
    assert arg.mode == 'generate_sentences', f'arg.mode must be set to generate_sentences'
    
    model = torch.load(arg.checkpoint_path)
    model.eval()
    with torch.no_grad():
        model.g
    exit_at = time.time()
    print(f'Time taken to train is {(exit_at - enter_at)/60} mins.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'generate_sentences'], required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--shuffle_train', action='store_true')
    parser.add_argument('--model', type=str, choices=['Bert2Bert'], required=True)
    parser.add_argument('--checkpoint_path', type=str, default='/tmp/checkpoint.pth')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_input', type=str, required=True)
    parser.add_argument('--development_input', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--toy_dataset', type=int, default=0)
    parser.add_argument('--input_string', type=str, default='subtable_str_plus_subtable_metadata_str', choices=['subtable_str_plus_subtable_metadata_str', 'raw_table'])
    parser.add_argument('--metric', type=str, default='BLEU', choices=['BLEU', 'ROUGE'])
    parser.add_argument('--task', type=str, default='TOTTO', choices=['TOTTO', 'cnn_dailymail'])
    arg = parser.parse_args()
    print(arg)
    if (arg.mode == 'train'):
        train(arg)
    print(f'Exiting....')

    
            

        





