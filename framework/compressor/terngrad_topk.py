#########################################################################################
# This is the first open-source hybrid algorithm between TernGrad and Top-k.            #
# This implementation was inspired by the open-source implementation in the             #
# "TernGrad: ternary gradients to reduce communication in distributed deep              #
# learning" paper:                                                                      #
#                       (https://github.com/wenwei202/terngrad)                         #
# Horovod's Gradient compression implementation:                                        #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)    #
# and GRACE open-source framework:                                                      #
#                   (https://github.com/sands-lab/grace)                                #
#########################################################################################

import torch
from compressor.compressor import Compressor


class TerngradTopKCompressor(Compressor):
    """
    This is a hybrid algorithm between TernGrad quantization and Top-k
    sparsification algorithms. It is implemented by sparsifying the gradints
    first, then quantizing the sparsified gradients. For details about the
    two original methods, please refer to the respective code.

    Args:
        compress_ratio: the ratio of the gradients to be kept.
        clip_const: a hyperparameter that decides on the gradients to be
                    clipped. It is task-dependant. For CIFAR-10/MNIST/ImageNet,
                    it was chosen to be 2.5 (as per the paper). For PTB Language
                    Model, values between 44 and 68 gave the best results.
    """

    def __init__(self, compress_ratio, clip_const):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.clip_const = clip_const
        self.total_compressed = 0
        self.total_original = 0

    def compress(self, tensor, name):
        """
        This method compresses the gradients by choosing the "compression_ratio"
        elements and transmits them along with their indices. Then it quantizes
        them using TernGrad's algorithm.

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

        quant_tensors, shape = quantize(values, self.clip_const)
        tensors = quant_tensors, indices

        ctx = tensor.numel(), tensor.size(), shape

        self.total_original += tensor.numel()
        self.total_compressed += values.numel() * (1/16) + indices.numel()
        compression_ratio = (self.total_original / self.total_compressed)

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
        numel, size, shape = ctx
        values, indices = tensors

        dequantized_tensors = dequantize(values, shape)
        dequant_tensors = dequantized_tensors, indices
        tensor_decompressed = desparsify(dequant_tensors, numel)

        return tensor_decompressed.view(size)


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


def quantize(tensor, clip_const):
    """
    This function quantizes the gradients as per TernGrad algorithm.

    Args:
        tensor: the tensor we need to quantize (after compensation by the
                residual memory).
        name: the name of the experiment (not used here).

    Returns:
        quantized_tensor: a tensor that contain the quantized gradients
                           and the mean value for the original gradients.
        shape: the shape of the original gradients' tensor.

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
    scalar = abs_gradient.max()

    # Step.3: getting the signs of all gradients and multiplying with the
    # scalar from Step.2.
    sign_gradient = gradient.sign() * scalar

    # Step.4: multiplying with a Bernoulli distribution (either 0 or 1).
    rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
    sign_gradient[rnd_sample >= abs_gradient] = 0
    ternarized_grads = sign_gradient.sign()     # {-1,0,+1}

    quantized_tensor = ternarized_grads.type(torch.int8), scalar.flatten()

    return quantized_tensor, shape


def dequantize(quantized_tensor, shape):
    """
    This function decompress the compressed tensor by restoring the original
    values from the compressed tensors.

    Args:
        tensor_compressed: a tensor that contain the ternarized gradients
                           and the scalar value for the original gradients.
        shape: the shape of the original gradients' tensor.

    Returns:
        dequantized_tensor: the decompressed tensor, in the same shape as
        the origonal gradients' tensor.
    """
    tensor_compressed, scalar = quantized_tensor
    sign = tensor_compressed.type(torch.float32)
    dequantized_tensor = sign * scalar
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
