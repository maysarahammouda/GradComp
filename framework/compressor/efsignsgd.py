#########################################################################################
# This implementation was inspired by the open-source implementation in the             #
# "Error Feedback Fixes SignSGD and other Gradient Compression Schemes" paper           #
#                       (https://github.com/epfml/error-feedback-SGD)                   #
# Horovod's Gradient compression implementation:                                        #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)    #
# and GRACE open-source framework:                                                      #
#                           (https://github.com/sands-lab/grace)                        #
#########################################################################################

import torch
from compressor.compressor import Compressor


class EFSignSGDCompressor(Compressor):
    """
    This quantization algorithms quantizes the 32-bit values in the original
    gradients into 1-bit values. It does that by keeping only the signs of the
    gradients. The special feature in the Error-Feedback SignSGD is the use of
    the residual memory which incorporates the the error made by the compression
    operator into the next step. Doing that guarantees the convergance of the
    algorithm to the same value as the SGD with the same convergance rate.

    Args:
        lr: learning rate (used only in the aggregation step - if needed).

    For more information:
    http://proceedings.mlr.press/v97/karimireddy19a.html
    """

    def __init__(self, lr):
        super().__init__(average=False)
        self.learning_rate = lr


    def compress(self, tensor, name):
        """
        This method compresses the gradients based on their signs. If the
        value of the gradient is greater than or equal to zero, the value will
        be quantized to +1 (or True), otherwise it will be quantized to 0 (or
        False). These values will be converted to +1 and -1 using the
        "decompress" method.

        Steps:
            1. Check the sign of the gradients and quantize the values to 1 or 0
                based on their sign.
            2. Save the mean value of the original tensor. This is important
                for the "decompress" function and for the residual memory.
                Without this, the convergence is not guaranteed.

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
        tensor_compressed, shape = quantize(tensor)

        compression_ratio = 32

        return tensor_compressed, shape, compression_ratio


    def decompress(self, tensor_compressed, shape):
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
        tensor_decompressed = dequantize(tensor_compressed, shape)

        return tensor_decompressed


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
