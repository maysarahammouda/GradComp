#########################################################################################
# This is the first open-source hybrid algorithm between EF-SignSGD and Top-k.          #
# This implementation was inspired by the open-source implementation in the             #
# "Error Feedback Fixes SignSGD and other Gradient Compression Schemes" paper           #
#                       (https://github.com/epfml/error-feedback-SGD)                   #
# Horovod's Gradient compression implementation:                                        #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)    #
# and GRACE open-source framework:                                                      #
#                   (https://github.com/sands-lab/grace)                                #
#########################################################################################

import torch
from compressor.compressor import Compressor


class EFSignTopKCompressor(Compressor):
    """
    This is a hybrid algorithm between EF-SignSGD quantization and Top-k
    sparsification algorithms. It is implemented by sparsifying the gradints
    first, then quantizing the sparsified gradients. For details about the
    two original methods, please refer to the respective code.

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
        This method compresses the gradients by choosing the "compression_ratio"
        elements and transmits them along with their indices. Then it quantizes
        then using EF-SignSGD algorithm.

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
        quant_tensors, shape = quantize(values)
        tensors = quant_tensors, indices

        ctx = tensor.numel(), tensor.size(), shape

        self.total_original += tensor.numel()
        self.total_compressed += values.numel()*(1/32) + indices.numel()
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


def quantize(tensor):
    """
    This function compresses the gradients based on their signs. If the
    value of the gradient is greater than or equal to zero, the value will
    be quantized to +1 (or True), otherwise it will be quantized to 0 (or
    False). These values will be converted to +1 and -1 using the
    "decompress" method.

    Args:
        tensor: the tensor we need to quantize (after compensation by the
                residual memory).
    Returns:
        tensor_compressed: a tensor that contain the quantized gradients
                           and the mean value for the original gradients.
        shape: the shape of the original gradients' tensor.
        compression_ratio: the amount of compression we get after compressing
                            the gradients.
    """
    shape = tensor.size()
    tensor = tensor.flatten()
    sign_encode = tensor >= 0
    mean = tensor.abs().mean()
    quantized_tensor = mean, sign_encode.type(torch.uint8)
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
        tensor_decompressed: the decompressed tensor, in the same shape as
        the origonal gradients' tensor.
    """
    mean, sign_encode = quantized_tensor
    sign_decode = sign_encode.type(torch.float32) * 2 - 1
    sign_decode = mean * sign_decode
    dequantized_tensor = sign_decode.view(shape)
    return dequantized_tensor


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
