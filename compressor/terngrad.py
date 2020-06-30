import torch
from compressor.compressor import Compressor


class TernGradCompressor(Compressor):
    """
    This quantization algorithms quantizes the gradients to a ternarty vector
    with values {-1,0,+1}.
    Args:
        clip_const: a hyperparameter that decides on the gradients to be
                    clipped. it is task-dependant. For CIFAR-10/MNIST/ImageNet, it was chosen to be 2.5 (as per the paper). In PTB LM, values between 9.7 and 19 gave the best results.
    """

    def __init__(self, clip_const):
        super().__init__()
        self.clip_const = clip_const

    def compress(self, tensor, name):
        """
        This function ternarizes the gradients (makes them take values {-1,0,-1}).
        Steps:
            1. Perform gradient clipping.
            2. Get the maximum norm (abs value) of all the gradients.
            3. Get the signs of all gradients, to keep the directions of the
                gradients, and multiply them with the scalars from Step.2.
            4. Multiply with a Bernoulli distribution (either 1 or 0 for each gradient).
        Args:
            tensor: the tensor we need to quantize.
            name: the name of the experiment (not used here).
        Returns:
            tensor_compressed: a tensor that contain the ternarized gradients
                               and the scalar value for these gradients.
            shape: the origional shape of the gradients' tensor.
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

        return compressed_tensor, shape

    def decompress(self, compressed_tensor, shape):
        """
        This function decompress  the gradients by restoring the origional values
        from the compressed tensors.
        Args:
            tensor_compressed: a tensor that contain the ternarized gradients
                               and the scalar value for these gradients.
            shape: the origional shape of the gradients' tensor.
        Returns:
            tensor_decompressed: the decompressed tensor, in the same shape as
            the origonal gradients' tensor.
        """
        tensor_compressed, scalar = compressed_tensor
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)
