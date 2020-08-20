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
    This sparsification algorithms chooses the most important gradients based
    on the relation between the magnitude and the gradients.

    For more information:
    https://arxiv.org/pdf/1802.06058.pdf

    We have found that this algorithm is totally inefficient when applied to
    large models/datasets. If you are able to create a more efficient one,
    please contact us.
    """

    def __init__(self, alpha, batch_size):
        super().__init__()
        self.alpha = alpha
        self.batch_size = batch_size
        self.total_compressed = 0
        self.total_original = 0

    def compress(self, tensor, name):
        """
        This method compresses the gradients by applying the Variance-Based
        sparsification algorithm.

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
        tensors = sparsify(tensor, self.alpha, self.batch_size)
        ctx = tensor.numel(), tensor.size()

        self.total_original += tensor.numel()
        self.total_compressed += tensors[0].numel()
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


def sparsify(tensor, alpha, batch_size):
    """
    This function performs "sparsification" for "tensor".

    Args:
        tensor: the tensor we need to sparsify.
        compress_ratio: the percentage of the number of elements we want to keep.

    Return:
        the values and indices for the choosen elements.
    """
    tensor = tensor.flatten()
    r,v = 0,0
    val,ind = [],[]
    gamma = 0.999

    for i in range(len(tensor)):
        r += tensor[i]/batch_size
        v += pow(tensor[i]/batch_size,2)
        if pow(r,2) > alpha*v:
            val.append(tensor[i])
            ind.append(i)
            r = 0
            v = 0
        else:
            v *= gamma

    values = torch.tensor(val)
    indices = torch.tensor(ind)
    return values, indices


def sparsify2(tensor, alpha, batch_size):
    """
    This function performs "sparsification" for "tensor".

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


def sparsify3(tensor, alpha, batch_size):
    """
    This function performs "sparsification" for "tensor".

    Args:
        tensor: the tensor we need to sparsify.
        compress_ratio: the percentage of the number of elements we want to keep.

    Return:
        the values and indices for the choosen elements.
    """
    tensor = tensor.flatten()
    r,v,m = 0,0,1
    ind = torch.zeros(tensor.numel())
    gamma = 0.999

    r = torch.cumsum(tensor/batch_size, dim=0)
    v = torch.cumsum(pow(tensor/batch_size,2), dim=0)

    for i in range(len(tensor)):
        if pow(r[m-1],2) > alpha*v[m-1]:
            ind[i] = 1
            r = torch.cumsum(tensor[i:]/batch_size, dim=0)
            v = torch.cumsum(pow(tensor[i:]/batch_size,2), dim=0)
            m = 0
        else:
            v[m-1] *= gamma
            m += 1

    indices = torch.where(ind>0)[0].flatten()
    values = tensor[indices]
    return values, indices


def sparsify4(tensor, alpha, batch_size):
    """
    This function performs "sparsification" for "tensor".

    Args:
        tensor: the tensor we need to sparsify.
        compress_ratio: the percentage of the number of elements we want to keep.

    Return:
        the values and indices for the choosen elements.
    """
    tensor = tensor.flatten()
    r,v,m = 0,0,1
    ind = torch.zeros(tensor.numel())
    gamma = 0.999

    r = torch.cumsum(tensor/batch_size, dim=0)
    v = torch.cumsum(pow(tensor/batch_size,2), dim=0)

    for i in range(len(tensor)):
        if pow(r[m-1],2) > alpha*v[m-1] or m == len(r) -1:
            ind[i] = 1
            r = torch.cumsum(tensor[i:i+100]/batch_size, dim=0) if len(tensor[i:]>100) else torch.cumsum(tensor[i:]/batch_size, dim=0)
            v = torch.cumsum(pow(tensor[i:i+100]/batch_size,2), dim=0) if len(tensor[i:]>100) else torch.cumsum(tensor[i:]/batch_size, dim=0)
            m = 1
        else:
            v[m-1] *= gamma
            m += 1

    indices = torch.where(ind>0)[0].flatten()
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
