#####################################################################################
# This code was adapted from:                                                       #
# https://github.com/salesforce/awd-lstm-lm/                                        #
#####################################################################################

import torch
import numpy as np
import argparse
import data_loader

def create_datasets (args, device):
    """
    This fuction reads the dataset and divides them into Train, Valid, and Test
    sets.
    """
    corpus = data_loader.Corpus(args.data)
    vocab_size = len(corpus.dictionary)
    train_data = batchify(corpus.train, args.batch_size, device)
    valid_data = batchify(corpus.valid, args.eval_batch_size, device)
    test_data = batchify(corpus.test, args.test_batch_size, device)
    return train_data, valid_data, test_data, vocab_size


def batchify(data, bsz, device):
    """
    This fuction divides the dataset into batches.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, args):
    """
    This fuction creates the input/output sequences.
    """
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len] # data.shape: torch.Size([seq_len, bsz])
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
