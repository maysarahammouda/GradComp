################################################################################
# This implementation was inspired by the implementation in GRACE open-source  #
# framework:                                                                   #
#                 (https://github.com/sands-lab/grace)                         #
################################################################################

from memory.Memory import Memory

class ResidualMemory(Memory):
    """
    This class implements the residual memory by subtracting the decompressed
    tensors' values from the original tensor's values so that these differences
    will not be forgotten (as explained in the literature in several papers).
    This implementation can handle parallel training. Here, num_workers is the
    number of parallel workers we are using in the experiment. The worker_id is
    the id for the current worker, which can take vales [0,num_workers-1].
    """

    def __init__(self, beta=1.0, gamma=1.0, num_workers=1):
        self.beta = beta
        self.gamma = gamma
        self.num_workers = num_workers
        self.all_residuals = {id:{} for id in range(num_workers)}

    def is_valid(self, worker_id):
        """
        This method checks whether the worker_id is valid or not.
        Returns:
            True (bool), if the worker_id is valid.
        Raises:
            RuntimeError, if the worker_id is not valid.
        """
        if(worker_id < 0 or worker_id > self.num_workers-1 or (not isinstance(worker_id, int)) ):
            raise RuntimeError('Invalid worker id! num_workers:{0}, current_workerId:{1}'.format(self.num_workers, worker_id))
        else:
            return True

    def compensate(self, tensor, name, worker_id=0):
        """
        This method update the tensor with the residuals from the previous step.
        It does that by adding the residual (multiplied by a scalar) to the
        tensor (multiplied by a scalar). Here we are keeping both scalars' values
        as 1.
        Args:
            tensor: the origional tensor, before compression.
            name: the parameter name.
            worker_id: the current worker id (batch % num_workers).
        Returns:
            tensor: the origional tensor after adding the residuals.
        """
        self.is_valid(worker_id)
        if name in self.all_residuals[worker_id]:
            tensor = self.beta * self.all_residuals[worker_id][name] + self.gamma * tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx, worker_id=0):
        """
        This method updates the residuals after the compression is done.
        It does that by subtracting the decompressed tensor from the origonal
        tensor. It then saves these residual into a dictionary.
        Args:
            tensor: the origional tensor, before compression.
            name: the parameter name.
            compressor: the compressor which is used in the experiment.
            tensor_compressed & ctx: the returns from the "compress" function.
            worker_id: the current worker id (batch % num_workers).
        Returns:
            Nothing.
        """
        self.is_valid(worker_id)
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        self.all_residuals[worker_id][name] = residual
