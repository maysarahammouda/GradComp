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

    def compress(self, tensor, name):
        """
        This function compresses the gradients by choosing the to "compression_ratio"
        elements and transmits them along with their indices.
        Args:
            tensor: the tensor we need to quantize.
            name: the name of the experiment (not used here).
        Returns:
            tensors: the compressed gradients' tensors.
            ctx: the context (the number of elements and the size of the compressed
                    gradients' tensor).
        """
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """
        This function decompress by filling empty slots with zeros and reshape
        back using the original shape.
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
    # why not   "values, indices = torch.topk(tensor.abs(), k)"  ??
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
