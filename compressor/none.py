from compressor.compressor import Compressor


class NoneCompressor(Compressor):
    """Default no-op compression."""

    def compress(self, tensor, name):
        return [tensor], None, 0

    def decompress(self, tensors, ctx):
        tensor, = tensors
        return tensor
