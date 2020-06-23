## This implpementation was inspired by PyTorch implpementation for SGD
## (https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py)
## and Horovod implementation for DistributedOptimizer
## (https://github.com/horovod/horovod/blob/31f1f700b8fa6d3b6df284e291e302593fbb4fa3/horovod/torch/__init__.py)

import torch
from torch.optim.optimizer import Optimizer, required
from compressor.topk import TopKCompressor
from compressor.randomk import RandomKCompressor
from compressor.onebit import OneBitCompressor
from compressor.none import NoneCompressor
from itertools import chain


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

    def __init__(self, params, compressor=NoneCompressor(), isNoneCompressor=False, num_workers=1, lr=required):
        self.compressor = compressor
        self.num_workers = num_workers
        self.isNoneCompressor = isNoneCompressor
        self.acc_comp_grads = {}
        self.lr = lr

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)

        super(SGD_Comp, self).__init__(params, defaults)


    def compress_grads(self):
        """
        This function compresses the gradients based on the compression algorithm
        chosen in self.compressor.
        Returns:
            comp_grads_tensors: two tensors containing the compressed gradients
                                and the corresponding indices.
            ctx: a context tensor that contains the number of elements and the
                 size of the origional gradients' tensor.
        """
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                grads.append(p.grad.view(-1))
        grads_tensor = torch.cat(grads)

        if self.isNoneCompressor:
            return grads_tensor
        else:
            comp_grads_tensors, ctx = self.compressor.compress(grads_tensor,"myGrads")
            comp_grads, indices = comp_grads_tensors
            return comp_grads_tensors, ctx


    @torch.no_grad()
    def step(self, closure=None):
        """
        This function performs a single optimization step.
        Args:
            closure (callable, optional): A closure that re-evaluates the model
                                            and returns the loss.
        Returns:
            loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        comp_grads_tensors, ctx = SGD_Comp.compress_grads(self)
        comp_grads, indices = comp_grads_tensors
        decomp_grads = self.compressor.decompress(comp_grads_tensors, ctx)

        c = []
        for group in self.param_groups:

            for p in group['params']:
                p = p.flatten()    # or  p = p.view(-1)
                torch.zeros_like(p).scatter()
                for i in p:
                    c.append(i)
                # torch.tensor(c).add_(decomp_grads, alpha=-self.lr)
        # for i, p in enumerate(c):
        #     p.add_(decomp_grads[i], alpha=-self.lr)
        for i, d_p in zip(indices,comp_grads):
            c[i].add_(d_p, alpha=-self.lr)

        # self.acc_comp_grads = {}  #for multiple workers
        return loss






# @torch.no_grad()
# def step(self, closure=None):
#     """
#     Performs a single optimization step.
#     Args:
#         closure (callable, optional): A closure that reevaluates the model
#             and returns the loss.
#     Returns:
#         loss
#     """
#     loss = None
#     if closure is not None:
#         with torch.enable_grad():
#             loss = closure()
#
#     for group in self.param_groups:
#         # a group for each batch --> we will have no. of batches groups
#         for i,p in enumerate(group['params']):
#             if p.grad is None:
#                 continue
#             d_p = p.grad
#             p.add_(d_p, alpha=-group['lr'])
#     return loss


# def compress_grads_workers(self):
#     grads = []
#     for group in self.param_groups:
#         for p in group['params']:
#             grads.append(p.grad.view(-1))
#     grads_tensor = torch.cat(grads)
#     print("grads_tensor:", grads_tensor)
#     # print("grads_tensor_max", max(abs(grads_tensor)))
#
#     # comp_grads_tensors is a list that contains two tensors: the
#     # compressed gradients and the indicies of these compressed gradients
#     if self.isNoneCompressor:
#         for index, grad in enumerate(grads_tensor):
#             self.acc_comp_grads[index] = grad
#
#     else:
#         comp_grads_tensors, ctx = self.compressor.compress(grads_tensor,"myGrads")
#         comp_grads, indices = comp_grads_tensors
#         # print("origional grads:", grads_tensor.shape)
#         # print("compressed_grads type:", type(comp_grads))
#         # print("compressed_grads len:",len(comp_grads))
#         # print("ctx:", ctx)
#         print("comp_grads:",comp_grads)
#         # print("indices:",indices)
#
#         for ind, grad in zip(indices, comp_grads):
#             # print(ind.item(),grad.item())
#             index = ind.item()
#             if index not in self.acc_comp_grads.keys():
#                 self.acc_comp_grads[index] = grad.item()
#                 print("case1:",len(self.acc_comp_grads),self.acc_comp_grads)
#             else:
#                 self.acc_comp_grads[index] += grad.item()
#                 print("case2:",len(self.acc_comp_grads),self.acc_comp_grads)
#         # decompressed_grads = self.compressor.decompress(compressed_grads, ctx)
#         # print("decompressed_grads type:", type(decompressed_grads))
#         # print("decompressed_grads shape:", decompressed_grads.shape)
#         # print("Compressor Type:",type(self.compressor))
#         # print("origional grads:", grads_tensor.shape)
#         # print("compressed grads:",len(self.acc_comp_grads))
#         # print("compressed grads:",self.acc_comp_grads)
#
#     # return self.acc_comp_grads
#     return
    # summed_tensor_compressed = HorovodAllreduce.apply(tensor_compressed, average, name, op)
    # return self.compressor.decompress(summed_tensor_compressed, ctx)
    # print((grads_tensor != 0).sum(dim=0))
    # print(type(self.param_groups))
    # print(self.param_groups[0])
    # print(decomp_grads.type)
