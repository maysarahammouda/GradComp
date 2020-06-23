
import torch
import numpy as np
import argparse
import data_loader

def create_datasets (args, device):
        corpus = data_loader.Corpus(args.data)
        vocab_size = len(corpus.dictionary)
        # print('Vocabluary size: {}'.format(vocab_size))
        train_data = batchify(corpus.train, args.batch_size, device)
        valid_data = batchify(corpus.valid, args.eval_batch_size, device)
        test_data = batchify(corpus.test, args.test_batch_size, device)
        return train_data, valid_data, test_data, vocab_size


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, args):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len] # data.shape: torch.Size([seq_len, bsz])
    target = source[i+1:i+1+seq_len].view(-1)
    # print("len(data):", len(data))
    # print('get_batch data shape', data.shape)
    return data, target
