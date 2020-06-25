import torch
from compressor.compressor import Compressor


class TernGradCompressor(Compressor):

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)   # Gradient clipping
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()         # equation.2 in the paper (St)

        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape


    def decompress(self, tensor_compressed, shape):
        """
        This function decompress  the gradients by restoring the origional values
        from the compressed tensors, which contain the terngrad gradients and the
        scalar values for these gradients, and the origional shape.
        """
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)
