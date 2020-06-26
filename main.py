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
from model_eval import train, evaluate
from batch_generation import create_datasets
from utils import get_num_parameters, save_model, repackage_hidden, str2bool
from utils import check_cuda, log_args

from optimizer import SGD_Comp
from memory.none import NoneMemory
from compressor.dgc import DgcCompressor
from memory.residual import ResidualMemory
from compressor.none import NoneCompressor
from compressor.topk import TopKCompressor
from compressor.randomk import RandomKCompressor
from compressor.onebit import OneBitCompressor
from compressor.terngrad import TernGradCompressor
from compressor.signsgd import SignSGDCompressor
from compressor.signum import SignumCompressor
from compressor.efsignsgd import EFSignSGDCompressor
from compressor.qsgd import QSGDCompressor

################################# Command Line Arguments  #################################

parser = argparse.ArgumentParser(description='A simple LSTM Language Model')
parser.add_argument('--data', type=str, default='../datasets/test', help='location of the data corpus')
parser.add_argument('--emb_size', type=int, default=200, help='size of word embeddings')
parser.add_argument('--num_hid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--init_lr', type=float, default=1.0, help='initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.0, help='learning rate decay factor')
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
parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
parser.add_argument('--compress_ratio', type=float, default=1.0, help='compress ratio for the compression techniques')
parser.add_argument('--compressor', type=str, help='the name of the compression technique')
parser.add_argument('--memory', type=str, help='the name of the memory technique')
args = parser.parse_args()

################################# Main Code #################################

if __name__ == '__main__':
    # Setting the random seed manually.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initializing wabdb
    wandb.init(name=args.exp_name, project=args.project_name)

    # Checking if there is any GPU available.
    check_cuda(args)
    device = torch.device("cuda" if args.use_gpu else "cpu")

    # Loading data and creating training, validation, and test data.
    train_data, valid_data, test_data, vocab_size = create_datasets(args, device)

    # Building the model.
    model = LSTM(vocab_size=vocab_size, batch_size=args.batch_size, embedding_size= args.emb_size,
                hidden_size=args.num_hid, num_layers=args.num_layers, dropout_rate=args.dropout, num_step=args.bptt)
    model.to(device)

    # Logging histograms of parameters and gradients values.
    wandb.watch(model, log="all")   #Valid options for the log argument are: "gradients", "parameters", "all", or None.

    # Learning rate configuration.
    lr = args.init_lr
    lr_decay_factor = 1 / (1+args.lr_decay)   # decay factor for learning rate
    m_flat_lr = 6.0           # number of epochs before decaying the learning rate

    criterion = nn.CrossEntropyLoss()   # criterion is default average by minibatch(size(0))

    # Choosing the compression algorithm.
    if args.compressor == "none":
        compressor = NoneCompressor()
    elif args.compressor == "topk":
        compressor = TopKCompressor(compress_ratio=args.compress_ratio)
    elif args.compressor == "randomk":
        compressor = RandomKCompressor(compress_ratio=args.compress_ratio)
    elif args.compressor == "onebit":
        compressor = OneBitCompressor()
    elif args.compressor == "dgc":
        compressor = DgcCompressor(compress_ratio=args.compress_ratio)
    elif args.compressor == "terngrad":
        compressor = TernGradCompressor()
    elif args.compressor == "threshold":
        compressor = ThresholdCompressor(threshold=0.1)
    elif args.compressor == "signsgd":
        compressor = SignSGDCompressor()
    elif args.compressor == "signum":
        compressor = SignumCompressor(momentum=0.1)
    elif args.compressor == "efsignsgd":
        compressor = EFSignSGDCompressor(lr=lr)
    elif args.compressor == "qsgd":
        compressor = QSGDCompressor(quantum_num=0.1)
    else:
        raise Exception("Please choose an appropriate compression algorithm...")

    # Choosing the memory technique.
    if args.memory == "none":
        memory = NoneMemory()
    elif args.memory == "residual":
        memory = ResidualMemory(n_worker=args.num_workers)
    else:
        raise Exception("Please choose an appropriate memory technique...")

    optimizer = SGD(model.parameters(), lr=lr)

    print("="*50)
    print("|"," "*18,"Training"," "*18,"|")
    print("="*50)

    # Running the model on the training and validation data
    for epoch in range(1, args.epochs + 1):
        lr_decay = lr_decay_factor ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay

        epoch_start_time = time.time()
        train(model, criterion, optimizer, vocab_size, train_data, epoch, lr, device, args,
                compressor, memory)
        evaluate(model, vocab_size, valid_data, criterion, epoch, epoch_start_time, args, False)

    print("="*50)
    print("|"," "*18,"Testing"," "*19,"|")
    print("="*50)

    # Running the model on the test data
    evaluate(model, vocab_size, test_data, criterion, epoch, epoch_start_time, args, True)

    # logging the number of parameters values to wandb
    get_num_parameters(model)

    # Logging all the command line arguments to wandb
    log_args(args)

    # Saving the model
    # model.save(os.path.join(wandb.run.dir, "model.h5"))

    print("\n======================== Done! ========================")
