## This implpementation was inspired by PyTorch implpementation for SGD
## (https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py)
## and Horovod implementation for DistributedOptimizer
## (https://github.com/horovod/horovod/blob/31f1f700b8fa6d3b6df284e291e302593fbb4fa3/horovod/torch/__init__.py)

import torch
# from .optimizer import Optimizer, required
from torch.optim.optimizer import Optimizer, required
from compressor.topk import TopKCompressor
from compressor.randomk import RandomKCompressor
from compressor.onebit import OneBitCompressor
from compressor.none import NoneCompressor

class SGD_Comp(Optimizer):
    """
    This class is a modefied version of the origional SGD optimizer (by PyTorch).
    It handles the gradient compression techniques and eliminates all unncessary
        parts.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """

    def __init__(self, params, compressor=NoneCompressor(), num_workers=1, lr=required):
        self.compressor = compressor
        self.acc_comp_grads = {}
        self.num_workers = num_workers

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)

        super(SGD_Comp, self).__init__(params, defaults)


    def compress_grads(self):
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                grads.append(p.grad.view(-1))
        grads_tensor = torch.cat(grads)
        # print("grads_tensor", max(abs(grads_tensor)))

        # comp_grads_tensors is a list that contains two tensors: the
        # compressed gradients and the indicies of these compressed gradients
        comp_grads_tensors, ctx = self.compressor.compress(grads_tensor,"myGrads")
        comp_grads, indices = comp_grads_tensors
        # print("compressed_grads type:", type(compressed_grads))
        # print("compressed_grads len:",len(compressed_grads[0]))
        # print("compressed_grads shape:",compressed_grads[1].shape)
        # print(comp_grads)
        # print(indices)

        for ind, grad in zip(indices, comp_grads):
            # print(ind.item(),grad.item())
            index = ind.item()
            if index not in self.acc_comp_grads.keys():
                self.acc_comp_grads[index] = grad.item()
                # print("case1:",len(self.acc_comp_grads))
            else:
                self.acc_comp_grads[index] += grad.item()
                # print("case2:",len(self.acc_comp_grads))
        # decompressed_grads = self.compressor.decompress(compressed_grads, ctx)
        # print("decompressed_grads type:", type(decompressed_grads))
        # print("decompressed_grads shape:", decompressed_grads.shape)
        # print("Compressor Type:",type(self.compressor))

        # print("compress grads:",self.acc_comp_grads)
        return self.acc_comp_grads
        # summed_tensor_compressed = HorovodAllreduce.apply(tensor_compressed, average, name, op)
        # return self.compressor.decompress(summed_tensor_compressed, ctx)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_p = []

        for group in self.param_groups:
            # a group for each batch --> we will have no. of batches groups
            for i,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                print((i+1),p.view(-1).shape)
                d_p = p.grad
                p.add_(d_p, alpha=-group['lr'])
                # print("p:",p.shape )
                # print("d_p:",d_p.shape)

                all_p.append(p.view(-1))
                all_param = torch.cat(all_p)

        # comp_grads = SGD_Comp.compress_grads(self)
        comp_grads = {k: v / self.num_workers for k, v in self.acc_comp_grads.items()}

        # print("compressed grads shape:", comp_grads)
        # print("indices shape:", indices.shape)
            # print("grads_tensor", grads_tensor.shape)
            # print("all_param:", all_param.shape)

        return loss
