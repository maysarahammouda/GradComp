#########################################################################################
# This implementation was inspired by Horovod's Gradient compression implementation:    #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)    #
# and GRACE open-source framework:                                                      #
#                   (https://github.com/sands-lab/grace)                                #
#########################################################################################

import torch
from compressor.compressor import Compressor


class VarianceBasedCompressor(Compressor):
    """
    This sparsification algorithms chooses the top (highest absolute magnitude)
    gradients and communicates them.

    Args:
        compress_ratio: the ratio of the gradients to be kept.
    """

    def __init__(self, alpha, batch_size):
        super().__init__()
        self.alpha = alpha
        self.batch_size = batch_size
        self.total_compressed = 0
        self.total_origional = 0

    def compress(self, tensor, name):
        """
        This function compresses the gradients by choosing the to "compression_ratio"
        elements and transmits them along with their indices.
        Args:
            tensor: the tensor we need to quantize.
            name: the name of the experiment (not used here).
        Returns:
            tensors: the compressed gradients' tensors.
            ctx: the context (the number of elements and the size of the origional
                    gradients' tensor).
            compression_ratio: the amount of compression we get after compressing
                                the gradients.
        """
        tensors = sparsify(tensor, self.alpha, self.batch_size)
        ctx = tensor.numel(), tensor.size()

        self.total_origional += tensor.numel()
        self.total_compressed += tensors[0].numel()
        compression_ratio = self.total_origional / self.total_compressed

        return tensors, ctx, compression_ratio


    def decompress(self, tensors, ctx):
        """
        This function decompress the compressed tensor by filling empty slots
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


def sparsify(tensor, alpha, batch_size):
    """
    This function performs "sparsification" for "tensor".
    It decides on the number of elements to keep based on the "compress_ratio".
    Args:
        tensor: the tensor we need to sparsify.
        compress_ratio: the percentage of the number of elements we want to keep.
    Return:
        the values and indices for the choosen elements.
    """
    tensor = tensor.flatten()
    r,v = 0,0
    ind = torch.zeros(tensor.numel())
    gamma = 0.999

    for i in range(len(tensor)):
        r += tensor[i]/batch_size
        v += pow(tensor[i]/batch_size,2)
        if pow(r,2) > alpha*v:
            ind[i] = 1
            r = 0
            v = 0
        else:
            v *= gamma

    indices = torch.where(ind>0)[0].flatten()
    values = tensor[indices]
    return values, indices


def desparsify(tensors, numel):
    """
    This function re-shapes the sparsified values into the same shape as the
    origional tensor. This would make dealing with these values easier.
    Args:
        tensor: the tensor we need to desparsify.
        numel: the total number of elements in the origional tensor.
    Returns:
        The desparsified tensor
    """
    values, indices = tensors
    tensor_desparsified = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_desparsified.scatter_(0, indices, values)
    return tensor_desparsified