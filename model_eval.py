import torch
import time
import math
import numpy as np
import argparse
from batch_generation import get_batch
from utils import repackage_hidden


def train(model, criterion, optimizer, vocab_size, train_data, epoch, lr, args):
    """
    This function runs the model on training data.
    It turns on the training mode, which in turn enables dropout.
    Args:
        model: the DL model defined in the model.py code.
        criterion: the loss function criterion defined in the main code.
        optimizer: the optimizer method chosen in the main code.
        vocab_size: the vocabulary size of the dataset (number of tokens).
        train_data: the dataset that contains the training data.
        epoch: the epoch number, to loop through the data.
        lr: the learning rate.
        args: the command line arguments from the main code.
    Returns:
        Nothing.
    """

    model.train()
    model.zero_grad()

    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)

    total_loss, num_seq, iters = 0.0, 0, 0

    num_fullSeq = (train_data.size(0) - 1) // args.bptt     # -1 for predicting the next word
    last_seqLen = (train_data.size(0) - 1) % args.bptt      # if last_seqLen=0 means all sequences' lengths = args.bptt

    num_seq = num_fullSeq if last_seqLen == 0 else num_fullSeq + 1

    last_update_worker = num_seq % args.nworker

    # print('number of full sequence', num_fullSeq)
    # print('last sequence length', last_seqLen)
    # print('number of sequence', num_seq)
    # print('last update worker', last_update_worker)

    log_interval = int(args.log_interval * args.nworker * args.batch_size)  # too much

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):     #range(start, stop, step) # or range(1, train_data.size(0))
        data, targets = get_batch(train_data, i, args)  # i = batch * bptt
        # print("\ndata:", data)
        # print("target:", targets)

        # Detaching the hidden state from how it was previously produced.
        # Otherwise, the model would try to backpropagat all the way to the start of the dataset.
        hidden = repackage_hidden(hidden)

        output, hidden = model(data, hidden)
        # print('output.shape:',output.view(-1, vocab_size).shape)

        # Not all workers could receive an input sequence
        current_nworker = last_update_worker if (last_update_worker != 0 and batch >= num_seq - last_update_worker) else args.nworker
        # print("\ncurrent_nworker", current_nworker)

        # The last worker may receive shorter sequence
        current_seqLen = last_seqLen if (batch == num_seq - 1 and last_seqLen > 0) else args.bptt
        # print("current_seqLen", current_seqLen)

        predictions = output.view(-1, vocab_size)
        loss = criterion(predictions, targets) / current_nworker

        total_loss += loss.item() * current_seqLen * current_nworker    # was origionally total_loss += loss.data.item() * args.bptt
        iters += current_seqLen     # was origonally iters += args.bptt

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)   # To prevent the exploding gradient problem.

        if (((batch + 1) % args.nworker == 0 and batch > 0) or (batch == num_seq - 1)):

            optimizer.step()
            model.zero_grad()

            # log training result
            _log_training_results (epoch, batch, lr, log_interval, num_fullSeq, num_seq, last_update_worker, total_loss, iters, start_time, args)


def evaluate(model, vocab_size, data_source, criterion, args):
    """
    This function evaluates the model on validation and test data.
    It turns on the evaluation mode, which in turn disables dropout.
    Args:
        model: the DL model defined in the model.py code.
        vocab_size: the vocabulary size of the dataset (number of tokens).
        data_source: the dataset to be evaluated (validation / test).
        criterion: the loss function criterion defined in the main code.
        args: the command line arguments from the main code.
    Returns:
        perplexity.
    """

    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(args.eval_batch_size)
    # hidden = model.init_hidden()

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * criterion(output_flat, targets).item() # len(data) = seq_len
            hidden = repackage_hidden(hidden)

    return total_loss / len(data_source)


def _log_training_results (epoch, batch, lr, log_interval, num_fullSeq, num_seq, last_update_worker, total_loss, iters, start_time, args ):
    """
    This is a helper function which helps in printing/logging the trainign results.
    It was created just to make the training function cleaner.
    """
    if ((batch+1) % log_interval== 0 and batch > 0) or (batch == num_fullSeq -1):

        # last update
        if (batch == num_seq - 1) and (last_update_worker != 0):
            cur_loss = total_loss / iters
            elapsed = time.time() - start_time
            train_ppl = math.exp(cur_loss)
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'
                    .format(epoch, batch, num_seq, lr, elapsed * 1000 / (log_interval - args.nworker + last_update_worker),
                    cur_loss, train_ppl))
            # logging the ppl values to wandb
            wandb.log({"Train perplexity": train_ppl})
        # normal log
        else:
            cur_loss = total_loss / iters
            elapsed = time.time() - start_time
            train_ppl = math.exp(cur_loss)
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'
                    .format(epoch, batch, num_seq, lr, elapsed * 1000 / log_interval, cur_loss, train_ppl))
            # logging the ppl values to wandb
            wandb.log({"Train perplexity": train_ppl})
        # total_loss = 0
        start_time = time.time()
