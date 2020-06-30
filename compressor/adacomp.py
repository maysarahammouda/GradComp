import torch
from compressor.compressor import Compressor


class AdaCompCompressor(Compressor):
    """
    This quantization algorithms quantizes the gradients to a ternarty vector
    with values {-1,0,+1}.
    Args:
        clip_const: a hyperparameter that decides on the gradients to be
                    clipped. it is task-dependant. For CIFAR-10/MNIST/ImageNet, it was chosen to be 2.5 (as per the paper). In PTB LM, values between 9.7 and 19 gave the best results.
    """

    def __init__(self, compensation_const):
        super().__init__()
        self.compensation_const = compensation_const

    def compress(self, grads, tensor, name):
        """
        This function ternarizes the gradients (makes them take values {-1,0,-1}).
        Steps:
            1. Perform gradient clipping.
            2. Get the maximum norm (abs value) of all the gradients.
            3. Get the signs of all gradients, to keep the directions of the
                gradients, and multiply them with the scalars from Step.2.
            4. Multiply with a Bernoulli distribution (either 1 or 0 for each gradient).
        Args:
            tensor: the tensor we need to quantize (after compernsation by the
                    residual memory -if applicable-).
            name: the name of the experiment (not used here).
        Returns:
            tensor_compressed: a tensor that contain the ternarized gradients
                               and the scalar value for these gradients.
            shape: the origional shape of the gradients' tensor.
        """
        ctx = tensor.numel(), tensor.size()
        grads = grads.flatten()
        tensor_G = tensor.flatten()
        tensor_H = tensor_G + self.compensation_const * grads

        # Step.1: getting the maximum norm of all gradients.
        abs_gradient = tensor_G.abs()
        g_max = abs_gradient.max()

        # Step.2:
        mask = tensor_H >= g_max
        compressed_tensor = tensor_H[mask]
        indices = torch.nonzero(mask)
        # print("compressed_tensor", compressed_tensor.size())
        # print("indices", indices.flatten().size())

        tensors = compressed_tensor, indices.flatten()

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
