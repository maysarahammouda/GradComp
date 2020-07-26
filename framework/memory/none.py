from memory.Memory import Memory


class NoneMemory(Memory):
    def compensate(self, tensor, name, worker_id=0):
        """Update the tensor with the residuals."""
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx, worker_id=0):
        """Update the residuals."""
        pass