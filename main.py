import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
import argparse
import time
import numpy as np
from utils import batch_generator, raw_data, get_num_parameters, save_model, repackage_hidden
from model import LSTM
import os
import datetime
import wandb
from torchsummary import summary
# import tqdm
# import math

# torch.nn.utils.parameters_to_vector(parameters)

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
parser.add_argument('--num_steps', type=int, default=35, help='number of LSTM steps')
parser.add_argument('--accumulation_steps', type=int, default=4, help='accumulation steps / n-batch / number of workers')
parser.add_argument('--dp_keep_prob', type=float, default=0.5, help='dropout *keep* probability')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--use_gpu', action='store_false', default=False, help='use GPU for training')
randomhash = ''.join(str(time.time()).split('.'))
# parser.add_argument('--save', type=str,  default=randomhash+'.pt', help='path to save the final model')
parser.add_argument('--save', type=str,  default="LSTM_model", help='path to save the final model')

args = parser.parse_args()

# Extracting the information from the arguments:
data = args.data
dataset_name = args.dataset_name
num_epochs = args.num_epochs
batch_size = args.batch_size
num_layers = args.num_layers
hidden_size = args.hidden_size
initial_lr = args.initial_lr
num_steps = args.num_steps
accumulation_steps = args.accumulation_steps
dp_keep_prob = args.dp_keep_prob
seed = args.seed
use_gpu = args.use_gpu
save = args.save

# batch_size = batch_size / accumulation_steps

# dataset_name = "test"
num_epochs = 1
batch_size = 20
accumulation_steps = 5
# hidden_size = 30
# num_steps = 2

################################# Functions' definitions  #################################

def run_epoch(model, data, is_train=False, lr=1.0):
    """
    This function runs the model on the given data.
    Args:
        model: the language model we want to use.
        data: the dataset we want to use. This can be a training, a validation, or a test dataset.
        is_train: if true, the fuction will train the model on the given dataset.
                  if fale, the function will just evaluate the model on the given dataset (without training).
        lr: the learning rate.
    Returns:
        perplexity.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps   # For visualization
    start_time = time.time()
    hidden = model.init_hidden()

    costs = 0.0
    iters = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    for batch_idx, (input, target) in enumerate(batch_generator(data, model.batch_size, model.num_steps)):
        # print(batch_idx)
        model.zero_grad()
        inputs = Variable(torch.from_numpy(input.astype(np.int64)).transpose(0, 1).contiguous())
        targets = Variable(torch.from_numpy(target.astype(np.int64)).transpose(0, 1).contiguous())
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)     # predictions = model(inputs)  # Forward pass

        labels = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))    # previously tt
        predictions = outputs.view(-1, model.vocab_size)

        loss = criterion(predictions, labels)    # loss = loss_function(predictions, labels)
        # loss = loss / accumulation_steps

        costs += float(loss.data) * model.num_steps
        iters += model.num_steps

        if is_train:
            # model.zero_grad()
            optimizer.zero_grad()  # set all weight grads from previous training iters to 0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            # optimizer.step()

            if (batch_idx+1) % accumulation_steps == 0:
                # print("batch_idx:",batch_idx+1)
                optimizer.step()
                model.zero_grad()

            if batch_idx % (epoch_size // 10) == 10:
                print("Percentage Done: {:2f}%    |  Perplexity: {:8.2f}     |   Speed: {:8.2f} wps".format(batch_idx * 100.0 / epoch_size, np.exp(costs / iters),
                                                           iters * model.batch_size / (time.time() - start_time)))
                # logging the loss values to wandb
                wandb.log({"loss": loss})
    return np.exp(costs / iters)

################################# Main Code  #################################

if __name__ == "__main__":
    torch.manual_seed(seed)
    np.random.seed(seed)
    wandb.init(config=args)

    raw_data = raw_data(data_path=data, prefix=dataset_name)
    train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
    vocab_size = len(word_to_id)
    print('Vocabluary size: {}'.format(vocab_size))

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

        train_p = run_epoch(model, train_data, True, lr)
        print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))

        valid_p = run_epoch(model, valid_data)
        print('Validation perplexity at epoch {}: {:8.2f}'.format(epoch, valid_p))

        # logging the ppl values to wandb
        wandb.log({"Train perplexity": train_p})
        wandb.log({"Validation perplexity": valid_p})

    print("="*50)
    print("|"," "*18,"Testing"," "*19,"|")
    print("="*50)

    model.batch_size = 1 # to make sure we process all the data
    test_p = run_epoch(model, test_data)
    print('Test Perplexity: {:8.2f}'.format(test_p))

    # logging the ppl values to wandb
    wandb.log({"Test perplexity": test_p})
    total_num_params, trainable_params, non_trainable_params = get_num_parameters(model)
    wandb.log({"Number of parameters": total_num_params})
    wandb.log({"Trainable Parameters": trainable_params})
    wandb.log({"Non-Trainable Parameters": non_trainable_params})
    # summary(model,input_size=(batch_size,num_steps,95))
    # Saving the model
    # save_model(save, model)

    print("\n======================== Done! ========================")
