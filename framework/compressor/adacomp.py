#########################################################################################
# This implementation was created from scratch with a minor inspiration from Horovod's  #
# Gradient compression implementation:                                                  #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)    #
# and GRACE open-source framework:                                                      #
#                           (https://github.com/sands-lab/grace)                        #
#########################################################################################

import torch
from compressor.compressor import Compressor


class AdaCompCompressor(Compressor):
    """
    This is an adaptive algorithm that can compress all types of layers without
    notable performance degradation. It universally adapts the compression rate
    based on the layer type, batch size, and the data available in the batch.
    Args:
        compensation_const: a hyperparameter that controls the amount of data
                            to be communicated.
    """

    def __init__(self, compensation_const):
        super().__init__()
        self.compensation_const = compensation_const
        self.total_compressed = 0
        self.total_origional = 0

    def compress(self, grads, tensor, name):
        """
        This function sparsifies the gradients as per the AdaComp algorithm.
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
            ctx: the context (the number of elements and the size of the origional
                    gradients' tensor).
            compression_ratio: the amount of compression we get after compressing
                                the gradients.
        """
        ctx = tensor.numel(), tensor.size()

        grads = grads.flatten()
        tensor_G = tensor.flatten()
        tensor_H = tensor_G + self.compensation_const * grads

        # Step.1: getting the maximum norm of all gradients.
        abs_gradient = tensor_G.abs()
        g_max = abs_gradient.max()

        # Step.2:
        mask = tensor_H.abs() >= g_max
        compressed_tensor = tensor_H[mask]  # << these might also be quantized ....
        indices = torch.nonzero(mask)

        tensors = compressed_tensor, indices.flatten()

        self.total_origional += tensor_G.numel()
        self.total_compressed += compressed_tensor.numel()
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