###########################################################################################
# This implementation was inspired by the Horovod's Gradient compression implementation:  #
# (https://github.com/horovod/horovod/tree/31f1f700b8fa6d3b6df284e291e302593fbb4fa3)      #
# and GRACE open-source framework:                                                        #
#                         (https://github.com/sands-lab/grace)                            #
###########################################################################################

from compressor.compressor import Compressor


class NoneCompressor(Compressor):
    """
    This code does no compression. It takes the tensor as input and returns
    the same tensor.
    This is to help us in making the framework generalized for all cases.
    """

    def compress(self, tensor, name):
        return [tensor], None, 0


    def decompress(self, tensors, ctx):
        tensor, = tensors
        return tensor
