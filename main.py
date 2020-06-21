import numpy as np
import argparse
import time
import math
import os
import wandb
import torch
import torch.nn as nn
from torch.optim import SGD

from model import LSTM
from utils import get_num_parameters, save_model, repackage_hidden, str2bool, check_cuda
from batch_generation import create_datasets
from model_eval import train, evaluate
from model import LSTM

from optimizer import SGD_Comp
from compressor.none import NoneCompressor
from compressor.topk import TopKCompressor
from compressor.randomk import RandomKCompressor
from compressor.onebit import OneBitCompressor

################################# Command Line Arguments  #################################

parser = argparse.ArgumentParser(description='A simple LSTM Language Model')
parser.add_argument('--data', type=str, default='../datasets/test', help='location of the data corpus')
parser.add_argument('--embsize', type=int, default=200, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--init_lr', type=float, default=1.0, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N', help='evaluation batch size')
parser.add_argument('--test_batch_size', type=int, default=10, metavar='N', help='test batch size')
parser.add_argument('--bptt', type=int, default=35, help='number of LSTM steps')
parser.add_argument('--dropout', type=float, default=1, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--use_gpu', type=str2bool ,default=False, help='use CUDA. When debug it is False.')
parser.add_argument('--log_interval', type=int, default=2, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='/saved_models/', help='path to save the final model')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--exp_name', type=str, help='name of the experiment')
parser.add_argument('--project_name', type=str, help='name of the project')
parser.add_argument('--nworker', type=int, default=1, help='number of workers')
args = parser.parse_args()

################################# Main Code #################################

if __name__ == '__main__':
    # Set the random seed manually.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initializing wabdb
    wandb.init(name=args.exp_name, project=args.project_name)

    # Check if there is any GPU available
    check_cuda(args)
    device = torch.device("cuda" if args.use_gpu else "cpu")

    # Load data and creat training, validation, and test data
    train_data, valid_data, test_data, vocab_size = create_datasets(args, device)

    # Build the model
    model = LSTM(vocab_size=vocab_size, batch_size=args.batch_size, embedding_size= args.embsize,
                hidden_size=args.nhid, num_layers=args.nlayers, dropout_rate=args.dropout, num_step=args.bptt)
    model.to(device)

    # To log histograms of parameters and gradients values
    wandb.watch(model, log="all")   #Valid options for the log argument are: "gradients", "parameters", "all", or None.

    # Learning rate configuration
    lr = args.init_lr
    lr_decay_factor = 1 / 1.2   # decay factor for learning rate
    m_flat_lr = 6.0             # number of epochs before decaying the learning rate

    criterion = nn.CrossEntropyLoss()   # criterion is default average by minibatch(size(0))
    optimizer = SGD(model.parameters(), lr=lr)

    print("="*50)
    print("|"," "*18,"Training"," "*18,"|")
    print("="*50)

    # Run the model on the training and validation data
    for epoch in range(1, args.epochs + 1):
        lr_decay = lr_decay_factor ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay

        epoch_start_time = time.time()
        train(model, criterion, optimizer, vocab_size, train_data, epoch, lr, args)

        # if 't0' in optimizer.param_groups[0]:
        #     tmp = {}
        #     for prm in model.parameters():
        #         tmp[prm] = prm.data.clone()
        #         prm.data = optimizer.state[prm]['ax'].clone()

        val_loss = evaluate(model, vocab_size, valid_data, criterion, args)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'
                .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)

        # for prm in model.parameters():
        #     prm.data = tmp[prm].clone()
        # logging the ppl values to wandb
        # wandb.log({"Train perplexity": train_ppl})
        # wandb.log({"Validation perplexity": valid_ppl})

    print("="*50)
    print("|"," "*18,"Testing"," "*19,"|")
    print("="*50)

    # Run the model on the test data
    test_loss = evaluate(model, vocab_size, test_data, criterion, args)
    print('-' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('-' * 89)

    # logging the ppl values to wandb
    # wandb.log({"Test perplexity": test_ppl})

    # logging the number of parameters values to wandb
    total_num_params, trainable_params, non_trainable_params = get_num_parameters(model)
    wandb.log({"Number of parameters": total_num_params})
    wandb.log({"Trainable Parameters": trainable_params})
    wandb.log({"Non-Trainable Parameters": non_trainable_params})

    print("\n======================== Done! ========================")
