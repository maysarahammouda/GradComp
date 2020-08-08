#########################################################################################
# This implementation was inspired by Horovod's Gradient compression implementation:    #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)    #
# and GRACE open-source framework:                                                      #
#                   (https://github.com/sands-lab/grace)                                #
#########################################################################################

import torch
from compressor.compressor import Compressor


class TerngradAdaCompCompressor(Compressor):
    """
    This sparsification algorithms chooses the top (highest absolute magnitude)
    gradients and communicates them.

    Args:
        compress_ratio: the ratio of the gradients to be kept.
    """

    def __init__(self, compensation_const, clip_const):
        super().__init__()
        self.compensation_const = compensation_const
        self.clip_const = clip_const
        self.total_compressed = 0
        self.total_original = 0

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
            ctx: the context (the number of elements and the size of the original
                    gradients' tensor).
            compression_ratio: the amount of compression we get after compressing
                                the gradients.
        """
        values, indices = sparsify(grads, tensor, self.compensation_const)
        quant_tensors, shape = quantize(values, self.clip_const)
        tensors = quant_tensors, indices.flatten()

        ctx = tensor.numel(), tensor.size(), shape

        self.total_original += tensor.numel()
        self.total_compressed += values.numel() * (1/16) + indices.numel()
        compression_ratio = (self.total_original / self.total_compressed)

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
        numel, size, shape = ctx
        values, indices = tensors

        dequantized_tensors = dequantize(values, shape)
        dequant_tensors = dequantized_tensors, indices
        tensor_decompressed = desparsify(dequant_tensors, numel)

        return tensor_decompressed.view(size)


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


def quantize(tensor, clip_const):
    """
    This method compresses the gradients based on their signs. If the
    value of the gradient is greater than or equal to zero, the value will
    be quantized to +1 (or True), otherwise it will be quantized to 0 (or
    False). These values will be converted to +1 and -1 using the
    "decompress" method.

    Args:
        tensor: the tensor we need to quantize (after compensation by the
                residual memory).
        name: the name of the experiment (not used here).

    Returns:
        tensor_compressed: a tensor that contain the quantized gradients
                           and the mean value for the original gradients.
        shape: the shape of the original gradients' tensor.
        compression_ratio: the amount of compression we get after compressing
                            the gradients.
    """
    shape = tensor.size()
    tensor = tensor.flatten()

    # Step.1: clipping the gradients.
    # equation(21) in the paper.
    std = (tensor - torch.mean(tensor)) ** 2
    std = torch.sqrt(torch.mean(std))   # the standard deviation of the gradients
    c = clip_const * std.item()
    gradient = torch.clamp(tensor, -c, c)

    # Step.2: getting the maximum norm of all gradients.
    # equation(2) in the paper (St)
    abs_gradient = gradient.abs()
    # print(abs_gradient)
    # print(abs_gradient.numel())
    scalar = abs_gradient.max() if abs_gradient.numel() > 0 else torch.tensor(0)
    print(scalar)

    # Step.3: getting the signs of all gradients and multiplying with the
    # scalar from Step.2.
    sign_gradient = gradient.sign() * scalar

    # Step.4: multiplying with a Bernoulli distribution (either 0 or 1).
    rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
    # print("rnd_sample:", rnd_sample.numel())
    sign_gradient[rnd_sample >= abs_gradient] = 0
    ternarized_grads = sign_gradient.sign()     # {-1,0,+1}

    quantized_tensor = ternarized_grads.type(torch.int8), scalar.flatten()

    return quantized_tensor, shape


def dequantize(quantized_tensor, shape):
    """
    This method decompress the compressed tensor by restoring the original
    values from the compressed tensors.

    Args:
        tensor_compressed: a tensor that contain the ternarized gradients
                           and the scalar value for the original gradients.
        shape: the shape of the original gradients' tensor.

    Returns:
        tensor_decompressed: the decompressed tensor, in the same shape as
        the origonal gradients' tensor.
    """
    tensor_compressed, scalar = quantized_tensor
    sign = tensor_compressed.type(torch.float32).cuda()
    dequantized_tensor = sign * scalar.cuda()
    return dequantized_tensor.view(shape)


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
