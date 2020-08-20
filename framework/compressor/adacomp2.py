#########################################################################################
# This is the first open-source implementation for AdaComp.                             #
# It was created from scratch with a minor inspiration from Horovod's Gradient          #
# compression implementation:                                                           #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)    #
# and GRACE open-source framework:                                                      #
#                           (https://github.com/sands-lab/grace)                        #
#########################################################################################

import torch
from compressor.compressor import Compressor


class AdaCompCompressor2(Compressor):
    """
    This is the same as AdaComp but here we added the gradient clipping as per
    TernGrad's implementation. This helps in clipping any outliers in the
    gradients.
    """

    def __init__(self, compensation_const):
        super().__init__()
        self.compensation_const = compensation_const
        self.total_compressed = 0
        self.total_original = 0

    def compress(self, grads, tensor, name):
        """
        This method sparsifies the gradients as per the AdaComp algorithm.

        Steps:
            1. Get the maximum norm (abs value) of all the gradients.
            2. Communicate only the values which satisfy the condition:
                |H(index)| >= g_max

        Args:
            grads: the gradients of the parameter group under consideration.
            tensor: the tensor we need to compress (after compensation by the
                    residual memory -if applicable-).
            name: the name of the experiment (not used here).

        Returns:
            tensors: the compressed gradients' tensors.
            ctx: the context (the number of elements and the size of the original
                    gradients' tensor).
            compression_ratio: the amount of compression we get after compressing
                                the gradients.
        """
        # Step.1: clipping the gradients.
        # equation(21) in the paper.
        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))   # the standard deviation of the gradients
        # c = self.clip_const * std.item()
        c = 60 * std.item()     # 60 here is the clip_const
        tensor = torch.clamp(tensor, -c, c)

        values, indices = sparsify(grads, tensor, self.compensation_const)
        tensors = values, indices.flatten()

        ctx = tensor.numel(), tensor.size()

        self.total_original += tensor.numel()
        self.total_compressed += values.numel() + indices.numel()
        compression_ratio = self.total_original / self.total_compressed

        return tensors, ctx, compression_ratio


    def decompress(self, tensors, ctx):
        """
        This method decompress the compressed tensor by filling empty slots
        with zeros and reshape back using the original shape.

        Args:
            tensors: the compressed gradients' tensors.
            ctx: the context (the number of elements and the size of the compressed
                    gradients' tensor).

        Returns:
            tensor_decompressed: the decompressed tensor, in the same shape as
            the origonal gradients' tensor.
        """
        numel, shape = ctx
        tensor_decompressed = desparsify(tensors, numel)
        return tensor_decompressed.view(shape)


def sparsify(grads, tensor, compensation_const):
    """
    This function performs "sparsification" for "tensor".
    It decides on the number of elements to keep based on the "compress_ratio".

    Args:
        tensor: the tensor we need to sparsify.
        compress_ratio: the percentage of the number of elements we want to keep.

    Return:
        the values and indices for the choosen elements.
    """
    grads = grads.flatten()
    tensor_G = tensor.flatten()
    tensor_H = tensor_G + compensation_const * grads

    # Step.1: getting the maximum norm of all gradients.
    abs_gradient = tensor_G.abs()
    g_max = abs_gradient.max()

    # Step.2: applying the sparsification threshold.
    mask = tensor_H.abs() >= g_max
    sparsified_tensor = tensor_H[mask]
    indices = torch.nonzero(mask)

    return sparsified_tensor, indices


def desparsify(tensors, numel):
    """
    This function re-shapes the sparsified values into the same shape as the
    original tensor. This would make dealing with these values easier.

    Args:
        tensor: the tensor we need to desparsify.
        numel: the total number of elements in the original tensor.

    Returns:
        The desparsified tensor
    """
    values, indices = tensors
    tensor_desparsified = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_desparsified.scatter_(0, indices, values)
    return tensor_desparsified
