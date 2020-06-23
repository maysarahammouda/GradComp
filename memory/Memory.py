# from https://github.com/sands-lab/grace/blob/555ca7bf18a921eeb7cd3d8ccbe77d9bf5dd8636/grace_dl/torch/memory/none.py

from abc import ABC, abstractmethod

class Memory(ABC):
    @abstractmethod
    def compensate(self, tensor, name, worker_id=0):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx, worker_id=0):
        """Update the residuals."""
        pass
