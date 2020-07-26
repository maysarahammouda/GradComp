#########################################################################################
# This implementation was inspired by the open-source implementation in the             #
# "TernGrad: ternary gradients to reduce communication in distributed deep              #
# learning" paper:                                                                      #
#                       (https://github.com/wenwei202/terngrad)                         #
# Horovod's Gradient compression implementation:                                        #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)    #
# and GRACE open-source framework:                                                      #
#                         (https://github.com/sands-lab/grace)                          #
#########################################################################################

import torch
from compressor.compressor import Compressor


class TernGradCompressor(Compressor):
    """
    This quantization algorithms quantizes the gradients to a ternarty vector
    with values {-1,0,+1}.
    This is an unbiased, it does not require the use of any memory to converge,
    unlike the biased algorithms which require the use of memory.

    Args:
        clip_const: a hyperparameter that decides on the gradients to be
                    clipped. it is task-dependant. For CIFAR-10/MNIST/ImageNet,
                    it was chosen to be 2.5 (as per the paper). In PTB LM,
                    values between 9.7 and 19 gave the best results.

    For more information:
    https://dl.acm.org/doi/abs/10.5555/3294771.3294915
    http://papers.nips.cc/paper/6749-terngrad-ternary-gradients-to-reduce-commun
    ication-in-distributed-deep-learning

    """

    def __init__(self, clip_const):
        super().__init__()
        self.clip_const = clip_const


    def compress(self, tensor, name):
        """
        This method ternarizes the gradients (makes them take values {-1,0,-1}).

        Steps:
            1. Perform gradient clipping.
            2. Get the maximum norm (abs value) of all the gradients.
            3. Get the signs of all gradients, to keep the directions of the
                gradients, and multiply them with the scalars from Step.2.
            4. Multiply with a Bernoulli distribution (either 1 or 0 for each
                gradient).

        Args:
            tensor: the tensor we need to quantize.
            name: the name of the experiment (not used here).

        Returns:
            compressed_tensor: a tensor that contain the ternarized gradients
                               and the scalar value for the origional gradients.
            shape: the shape of the origional gradients' tensor.
            compression_ratio: the amount of compression we get after compressing
                                the gradients.
        """
        shape = tensor.size()
        tensor = tensor.flatten()

        # Step.1: clipping the gradients.
        # equation(21) in the paper.
        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))   # the standard deviation of the gradients
        c = self.clip_const * std.item()
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

        compressed_tensor = ternarized_grads.type(torch.int8), scalar.flatten()

        compression_ratio = 16

        return compressed_tensor, shape, compression_ratio


    def decompress(self, compressed_tensor, shape):
        """
        This method decompress the compressed tensor by restoring an estimation
        of the origional values from the compressed tensors using the signs and
        the mean value of the gradients.

        Args:
            tensor_compressed: a tensor that contain the quantized gradients
                               and the mean value for the origional gradients.
            shape: the shape of the origional gradients' tensor.

        Returns:
            tensor_decompressed: the decompressed tensor, in the same shape as
            the origonal gradients' tensor.
        """
        tensor_compressed, scalar = compressed_tensor
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)
