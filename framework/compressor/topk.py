#########################################################################################
# This implementation was inspired by Horovod's Gradient compression implementation:    #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)    #
# and GRACE open-source framework:                                                      #
#                   (https://github.com/sands-lab/grace)                                #
#########################################################################################

import torch
from compressor.compressor import Compressor


class TopKCompressor(Compressor):
    """
    This sparsification algorithms chooses the top (highest absolute magnitude)
    gradients and communicates them.

    Args:
        compress_ratio: the ratio of the gradients to be kept.
    """

    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.total_compressed = 0
        self.total_original = 0

    def compress(self, tensor, name):
        """
        This function compresses the gradients by choosing the to "compression_ratio"
        elements and transmits them along with their indices.
        Args:
            tensor: the tensor we need to quantize.
            name: the name of the experiment (not used here).
        Returns:
            tensors: the compressed gradients' tensors.
            ctx: the context (the number of elements and the size of the original
                    gradients' tensor).
            compression_ratio: the amount of compression we get after compressing
                                the gradients.
        """
        values, indices = sparsify(tensor, self.compress_ratio)
        tensors = values, indices.flatten()
        ctx = tensor.numel(), tensor.size()

        self.total_original += tensor.numel()
        self.total_compressed += values.numel() + indices.numel()
        compression_ratio = self.total_original / self.total_compressed

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


def sparsify(tensor, compress_ratio):
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
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k)
    values = tensor[indices]
    return values, indices


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
