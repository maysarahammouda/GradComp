from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import collections
import numpy as np
import datetime
import torch
import re

# import pprint


def get_num_parameters(model):
    """
    This function calculates the number of parametes in the model.
    It takes the model as an argument and retuens the number of parameters.
    """
    total_num_params = 0
    trainable_params = 0
    non_trainable_params = 0
    for param in model.parameters():
        total_num_params += np.prod(param.shape)
        if param.requires_grad_:
            trainable_params += np.prod(param.shape)
        non_trainable_params = total_num_params - trainable_params
    return total_num_params, trainable_params, non_trainable_params

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print name, param.data


def save_model(save, model):
    """
    This function saves the model into the specified folder.
    It takes the model as an argument and retuens nothing.
    """
    with open(save, 'wb') as f:
        dt_string = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace("/","_").replace(" ","_").replace(":","_") + ".h5"
        torch.save(model.state_dict(), f=os.path.join("saved_models_inference", dt_string))
        torch.save(model, f=os.path.join("saved_models", dt_string))
        print("\nThe model has been saved to the saved_models folder!")
        return


def repackage_hidden(hidden):
    """
    This function wraps hidden states in new Variables, to detach them from their history.
    """
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(h) for h in hidden)


def _read_words(filename):
    """
    This function reads the words from a given file.
    Args:
        filename: the file we need to parse.
    Returns:
        The words available in the file. It also adds <eos> at the end of each sentence.
    """
    # with io.open(filename, 'r', encoding='utf-8')as f:
    #     return f.read().replace("\n", "<eos>").split()
    with open(filename, 'r', encoding='utf-8') as f:
        tokens = 0
        words = []
        for line in f:
            tokens = line.lower().split() + ['<eos>']
            for token in tokens:
                words.append(token)
        return words
        # return re.findall('[a-zA-Z0-9]+', f.read().lower().split())


def _build_vocab(filename):
    """
    This function builds the vocabulary from a given file.
    Args:
        filename: the file we need to parse.
    Returns:
        Two dictionaries: word_to_id & id_to_word, which link each word to a unique id.
    """
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))

    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    """
    This function converts a given file to word ids.
    Args:
        filename: the file we need to parse.
        word_to_id: the dictionary that links each word to its unique id.
    Returns:
        A list of the corresponding ids to all the words in the given file.
    """
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def raw_data(data_path=None, prefix="ptb"):
    """
    This function Loads the raw data from data directory "data_path".
    It reads the text files, converts strings to integer ids, and performs
    mini-batching of the inputs.
    Args:
        data_path: string path to the directory where the datasets are located.
    Returns:
        tuple (train_data, valid_data, test_data, vocabulary) where each of
        the data objects can be passed to batch_generator function.
        It also returns the word_to_id and id_to_word dictionaries.
    """
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_to_word = _build_vocab(train_path)

    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    return train_data, valid_data, test_data, word_to_id, id_to_word


def generate_batch(raw_data, batch_size, num_steps, num_workers):
    """
    This function iterates on the raw data and generates batch_size pointers,
    which allows minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from the raw_data function:
                train_data, valid_data, or test_data.
        batch_size: the batch size (int).
        num_steps: the number of unrolls (int).
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the right
        by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    eff_batch_size = batch_size * num_workers
    batch_len = data_len // eff_batch_size

    epoch_size = ((data_len // eff_batch_size - 1) // num_steps) * num_workers
    print("Epoch Size= ", epoch_size)

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[i*batch_size*num_steps : (i+1)*batch_size*num_steps]
        x = np.reshape(x,(num_steps, batch_size))
        # print("x", x)
        y = data[i*batch_size*num_steps+batch_size : (i+1)*batch_size*num_steps+batch_size]
        y = np.reshape(y,(num_steps, batch_size))
        # print("y", y)

        yield (x, y)


def batch_generator(raw_data, batch_size, num_steps):
    """
    This function iterates on the raw data and generates batch_size pointers,
    which allows minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from the raw_data function:
                train_data, valid_data, or test_data.
        batch_size: the batch size (int).
        num_steps: the number of unrolls (int).
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the right
        by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    # print("raw_data:", raw_data)

    data_len = len(raw_data)
    # print("len of raw data:", data_len)
    batch_len = data_len // batch_size
    # print("batch_len:", batch_len)
    # print("batch_size:", batch_size)
    # print("num_steps:",num_steps)

    data = np.zeros([batch_size, batch_len], dtype=np.int32)

    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    # print(data)

    epoch_size = (batch_len - 1) // num_steps
    # print("Epoch Size= ", epoch_size)

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        # print("x", x)
        # print(x.shape)
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        # print("y", y)
        # print(y.shape)

        yield (x, y)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    # if args.cuda:
    #     data = data.cuda()
    return data


def get_batch(args, source, i, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else seq_len, len(source) - 1 - i)
    data = source[i : i+seq_len]
    print("\ninputs",data)
    target = source[i+1 : i+1+seq_len].view(-1)
    print("labels",target)
    return data, target


# def batchify_dan(data, device, args, is_test=False):
def batchify_dan(data, bsz, args, is_test=False):
    if is_test:
        bsz = args.batch_size_test
    else:
        bsz = args.batch_size_train

    nbatch = data.size(0) // bsz

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    x = data.narrow(0, 0, nbatch * bsz)
    x = x.view(bsz, -1).t().contiguous()
    return x
    # return x.to(device)


def get_batch_dan(source, batch_idx, worker, args):
    """Returns the batch starting from position batch_idx."""
    i = batch_idx
    bsz = args.batch_size_train

    seq_length = min(args.seq_length, len(source) - 1 - i)
    print("seq_length:", seq_length)
    print("i", i)
    print(source)
    data = source[i : i+seq_length]
    print("\ninputs",data)
    target = source[i+1 : i+1+seq_length]
    print("labels",target)

    # chunch the data along batch dim, retrieving a chunch indexed by worker
    data_split = data[:,(bsz//args.num_workers)*worker : (bsz//args.num_workers)*(worker+1)]
    target_split = target[:,(bsz//args.num_workers)*worker : (bsz//args.num_workers)*(worker+1)]
    print(data_split)
    print(target_split)
    return data_split, target_split


def generate_batch2(raw_data, batch_size, num_steps, num_workers):
    """
    This function iterates on the raw data and generates batch_size pointers,
    which allows minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from the raw_data function:
                train_data, valid_data, or test_data.
        batch_size: the batch size (int).
        num_steps: the number of unrolls (int).
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the right
        by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    print("raw_data:", raw_data)

    data_len = len(raw_data)
    # print("len of raw data:", data_len)

    eff_batch_size = batch_size * num_workers
    batch_len = data_len // eff_batch_size
    # print("batch_len:", batch_len)
    # print("batch_size:", batch_size)
    # print("num_steps:",num_steps)

    data = np.zeros([eff_batch_size*num_steps, batch_len//num_steps], dtype=np.int32)
    print("data.shape:", data.shape)

    for i in range(eff_batch_size*num_steps):
        data[i] = raw_data[batch_len // num_steps * i:batch_len // num_steps * (i + 1)]
    print("data:",data)

    epoch_size = (batch_len - 1) // num_steps
    print("Epoch Size= ", epoch_size)

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        for j in range(num_workers):
            x = data[j*batch_size:(j+1)*batch_size, i*num_steps:(i+1)*num_steps]
            print("x", x)
            # print(type(x))
            # print(x.shape)
            y = data[j*batch_size:(j+1)*batch_size, i*num_steps+1:(i+1)*num_steps+1]
            print("y", y)
            # print(type(y))
            # print(y.shape)

            yield (x, y)


def get_batch_di(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len] # data.shape: torch.Size([seq_len, bsz])
    target = source[i+1:i+1+seq_len].view(-1)
    # print("len(data):", len(data))
    # print('get_batch data shape', data.shape)
    return data, target

# train_data, valid_data, test_data, word_to_id, id_to_word = raw_data("datasets","ptb")

## To print the "yield" output
# f = batch_generator(valid_data, 50 , 25)
# for value in f:
#     print(value, "\n\n")

## Pretty Print
# pp = pprint.PrettyPrinter()
# pp.pprint(words)

# ==============================================================================
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================