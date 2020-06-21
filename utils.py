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


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_cuda(args):
    if torch.cuda.is_available():
        if not args.use_gpu:
            print("You have a CUDA device but run on cpu.")
        else:
            print("You have a CUDA device and run on gpu.")
    else:
        print("You do not have a CUDA device and run on cpu.")
