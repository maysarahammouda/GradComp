from memory.Memory import Memory

class ResidualMemory(Memory):
    '''
    n_worker: the number of workers used to simulate parallel. Default is 1.
    worker_id: start from 0. Valid worker_id: 0 ,..., n_worker-1
    '''
    def __init__(self, beta=1.0, gamma=1.0, n_worker=1):
        self.beta = beta
        self.gamma = gamma
        self.n_worker = n_worker
        self.all_residuals = {id:{} for id in range(n_worker)}

    #check whether worker_id is valid
    def is_valid(self, worker_id):
        if(worker_id < 0 or worker_id > self.n_worker-1 or (not isinstance(worker_id, int)) ):
            raise RuntimeError('Invalid worker id!!! n_worker:{0}, current_workerId:{1}'.format(self.n_worker, worker_id))
        else:
            return True

    def compensate(self, tensor, name, worker_id=0):
        """Update the tensor with the residuals."""
        self.is_valid(worker_id)
        if name in self.all_residuals[worker_id]:
            tensor = self.beta * self.all_residuals[worker_id][name] + self.gamma * tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx, worker_id=0):
        """Update the residuals."""
        self.is_valid(worker_id)
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        self.all_residuals[worker_id][name] = residual
