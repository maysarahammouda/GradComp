import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from optimizer import SGD_Comp
import argparse
import time
import numpy as np
from utils import batch_generator, raw_data, get_num_parameters, save_model, repackage_hidden, generate_batch, generate_batch_, generate_batch2
from utils import batchify, get_batch
from model import LSTM
import os
import datetime
import wandb
from compressor.topk import TopKCompressor
from compressor.randomk import RandomKCompressor
from compressor.onebit import OneBitCompressor
from compressor.none import NoneCompressor
import data_load
# from torchsummary import summary
# import tqdm
# import math

################################# Command Line Arguments  #################################

# Reading the arguments from the command line
parser = argparse.ArgumentParser(description='A simple LSTM Language Model')
parser.add_argument('--data', type=str, default='datasets', help='location of the data corpus')
parser.add_argument('--dataset_name', type=str, default='ptb', help='name of the dataset')
parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--hidden_size', type=int, default=650, help='number of hidden units per layer (size of word embeddings)')
parser.add_argument('--initial_lr', type=float, default=1.0, help='initial learning rate')
parser.add_argument('--bptt', type=int, default=35, help='number of LSTM steps / bptt parameter')
parser.add_argument('--num_workers', type=int, default=4, help='accumulation steps / n-batch / number of workers')
parser.add_argument('--dp_keep_prob', type=float, default=0.5, help='dropout *keep* probability')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--use_gpu', action='store_false', default=False, help='use GPU for training')
randomhash = ''.join(str(time.time()).split('.'))
# parser.add_argument('--save', type=str,  default=randomhash+'.pt', help='path to save the final model')
parser.add_argument('--save', type=str,  default="LSTM_model", help='path to save the final model')
# parser.add_argument('--num_steps', type=int, default=2, help='bptt parameter')
args = parser.parse_args()

# Extracting the information from the arguments:
data = args.data
dataset_name = args.dataset_name
num_epochs = args.num_epochs
batch_size = args.batch_size
num_layers = args.num_layers
hidden_size = args.hidden_size
initial_lr = args.initial_lr
# num_steps = args.num_steps
num_workers = args.num_workers
dp_keep_prob = args.dp_keep_prob
seed = args.seed
use_gpu = args.use_gpu
save = args.save
bptt = args.bptt

dataset_name = "test"

if dataset_name == "ptb":
    num_epochs = 1
    batch_size = 20
    num_workers = 1
    log_interval = 10
    dp_keep_prob = 1
    hidden_size = 400
    initial_lr = 5.0
    # num_steps = 35

if dataset_name == "test":
    num_epochs = 1
    batch_size = 4
    num_workers = 2
    hidden_size = 30
    bptt = 2
    num_steps = 2
    log_interval = 1
    dp_keep_prob = 1
    initial_lr = 7.0

################################# Functions' definitions  #################################

def run_epoch(model, data, is_train=False, lr=1.0):
    """
    This function runs one epoch of the model on the given data.
    Args:
        model: the language model we want to use.
        data: the dataset we want to use. This can be a training, a validation, or a test dataset.
        is_train: if true, the fuction will train the model on the given dataset.
                  if fale, the function will just evaluate the model on the given dataset (without training).
        lr: the learning rate.
    Returns:
        perplexity.
    """
    # print(batch_size)
    # print(num_workers)
    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD_Comp(model.parameters(), compressor=TopKCompressor(compress_ratio=0.001), num_workers=num_workers, lr=lr)
    optimizer = SGD(model.parameters(), lr=lr)

    if is_train:
        model.train()
    else:
        model.eval()

    epoch_size = ((len(data) // model.batch_size) - 1) // args.bptt   # For visualization
    start_time = time.time()
    hidden = model.init_hidden()

    costs, iters = 0.0, 0

    for batch, i in enumerate(range(0, data.size(0) - 1, bptt)):
        inputs, targets = get_batch(args, data, i, bptt)
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)

        loss = criterion(outputs.view(-1, vocab_size), targets)
        # print("predictions",outputs.view(-1, vocab_size))
        costs += loss.item() * model.num_steps
        iters += model.num_steps

        if is_train:
            # print("\nbatch##", batch_idx+1)
            # model.zero_grad()
            # optimizer.zero_grad()  # set all weight grads from previous training iters to 0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            # optimizer.compress_grads()

            if (batch+1) % num_workers == 0:
                # print("\n***batch_idx -- n workers***:",batch_idx+1)
                optimizer.step()
                model.zero_grad()

            # if batch % (epoch_size // log_interval) == log_interval:    #### 10   10
                # print("Percentage Done: {:2f}%    |  Perplexity: {:8.2f}     |   Speed: {:8.2f} wps".format(
                # batch_idx * 100.0 / epoch_size, np.exp(costs / iters), iters * model.batch_size / (time.time() - start_time)))
                # logging the loss values to wandb
                # wandb.log({"loss": loss})
        # break
    return np.exp(costs / iters)

################################# Main Code  #################################

if __name__ == "__main__":
    torch.manual_seed(seed)
    np.random.seed(seed)
    wandb.init(config=args)

    corpus = data_load.Corpus(args.data)

    eval_batch_size = batch_size
    test_batch_size = 1

    train_data = batchify(corpus.train, batch_size, args)
    valid_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    print("Number of tokens:")
    print("Train: ", len(corpus.train))
    print("Valid: ", len(corpus.valid))
    print("Test:  ", len(corpus.test))

    vocab_size = len(corpus.dictionary)
    print("Vocab size:  {}".format(vocab_size))

    # raw_data = raw_data(data_path=data, prefix=dataset_name)
    # train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
    # vocab_size = len(word_to_id)
    # print('Vocabluary size: {}'.format(vocab_size))

    model = LSTM(embedding_dim=hidden_size, num_steps=num_steps, batch_size=batch_size,
                  vocab_size=vocab_size, num_layers=num_layers, dp_keep_prob=dp_keep_prob)
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)

    wandb.watch(model, log="all")   #Valid options for the log argument are: "gradients", "parameters", "all", or None.

    lr = initial_lr
    lr_decay_factor = 1 / 1.2  # decay factor for learning rate (was 1.15)
    m_flat_lr = 6.0    # number of epochs before decaying the learning rate (was 14)

    print("="*50)
    print("|"," "*18,"Training"," "*18,"|")
    print("="*50)

    for epoch in range(num_epochs):
        lr_decay = lr_decay_factor ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay

        train_ppl = run_epoch(model, train_data, True, lr)
        print('\nTrain perplexity at epoch {}: {:8.2f}'.format(epoch, train_ppl))

    #     valid_ppl = run_epoch(model, valid_data)
    #     print('\nValidation perplexity at epoch {}: {:8.2f}'.format(epoch, valid_ppl))
    #
    #     #logging the ppl values to wandb
    #     wandb.log({"Train perplexity": train_ppl})
    #     wandb.log({"Validation perplexity": valid_ppl})
    #
    # print("="*50)
    # print("|"," "*18,"Testing"," "*19,"|")
    # print("="*50)
    #
    # model.batch_size = 1 # to make sure we process all the data
    # test_ppl = run_epoch(model, test_data)
    # print('\nTest Perplexity: {:8.2f}'.format(test_ppl))
    #
    # # logging the ppl values to wandb
    # wandb.log({"Test perplexity": test_ppl})
    # total_num_params, trainable_params, non_trainable_params = get_num_parameters(model)
    # wandb.log({"Number of parameters": total_num_params})
    # wandb.log({"Trainable Parameters": trainable_params})
    # wandb.log({"Non-Trainable Parameters": non_trainable_params})

    # summary(model,input_size=(batch_size,num_steps,95))

    ## Saving the model
    # save_model(save, model)

    print("\n======================== Done! ========================")
