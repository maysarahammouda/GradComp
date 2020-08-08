import torch

from compressor.compressor import Compressor


class RandomKCompressor(Compressor):
    """
    This sparsification algorithms chooses some random gradients and communicates them.
    """

    def __init__(self, compress_ratio):
        super().__init__()
        self.global_step = 0
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        """
        This function uses Python Random libraries to compress by generating
        a list of indices to be transmitted.
        """
        h = sum(bytes(name, encoding='utf8'), self.global_step)
        self.global_step += 1
        torch.manual_seed(h)
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """
        This function decompress by filling empty slots with zeros and reshape
        back using the original shape.
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
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))
    indices = torch.randperm(numel, device=tensor.device)[:k]
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
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed
