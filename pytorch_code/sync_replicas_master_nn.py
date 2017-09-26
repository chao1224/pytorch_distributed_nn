from __future__ import print_function
import time

from mpi4py import MPI

from nn import NN_Trainer

STEP_START_ = 1

MAX_NUM_ITERATIONS = 1000000


def update_params_dist_version(layer, avg_grad, learning_rate):
    '''
    update the network layer by layer
    '''
    shape_tmp = layer.get_shape
    layer.W -= learning_rate * avg_grad[0:shape_tmp[0], :]
    layer.b -= learning_rate * avg_grad[shape_tmp[0], :]

class GradientAccumulator(object):
	'''a simple class to implement gradient aggregator like the `Conditional Accumulators` in tensorflow'''
	def __init__(self, module, num_worker):
		# we will update this counter dynamically during the training process
		# the length of this counter should be number of fc layers in the network
		model_length = len(module.state_dict())
		self.gradient_aggregate_counter = [0] * model_length
		self.model_index_range = [i for i in range(model_length)]

		# we used list to contain gradients of layers
		self.gradient_aggregator = []

		# TODO(hwang): we do not need to allocate so many space here since we need to aggregate the gradient
		# into each slot
		for param_idx,(key_name, param) in enumerate(module.state_dict().items()):
			tmp_aggregator = []
			for worker_idx in range(num_worker):
				tmp_aggregator.append(np.zeros((param.size())))
			# initialize the gradient aggragator
			self.gradient_aggregator.append(tmp_aggregator)

	def meset_everything(self):
		self._meset_grad_counter()
		self._meset_grad_aggregator()

	def _meset_grad_counter(self):
		self.gradient_aggregate_counter = [0 for _ in self.gradient_aggregate_counter]

	def _meset_grad_aggregator(self):
		'''reset the buffers in grad accumulator, not sure if this is necessary'''
		for i, tmp_aggregator in enumerate(self.gradient_aggregator):
			for j, buf in enumerate(tmp_aggregator):
				self.gradient_aggregator[i][j] = np.zeros(self.gradient_aggregator[i][j].shape)



class SyncReplicasMaster_NN(NN_Trainer):
	def __init__(self, comm, **kwargs):
		'''master node here, no rank needed since the rank will always be 0 for master node'''
		self.comm = comm   # get MPI communicator object
		self.world_size = comm.Get_size() # total number of processes
		self.cur_step = STEP_START_
		self.lr = kwargs['learning_rate']
		self.momentum = kwargs['momentum']

		self._num_grad_to_collect = self.world_size - 1
		# used to aggregate tmp gradients, the length is the same as # of fc layer 
		self._grad_aggregate_buffer = []
		self._model_shapes = []

	def build_model(self, num_layers, layer_config):
		# build network
        self.network=LeNet()
        # TODO(hwang): make sure this is useful
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        # assign a gradient accumulator to collect gradients from workers
        self.grad_accumulator = GradientAccumulator(module=self.network, num_worker=self.world_size-1)
        self.init_model_shapes()

	def train(self, training_set, validation_set):
		# the first step we need to do here is to sync fetch the inital worl_step from the parameter server
		# we still need to make sure the value we fetched from parameter server is 1
		# please note that step is start from one here
		self.async_bcast_step()

		# fake test here:
		for i in range(1, MAX_NUM_ITERATIONS):
			enough_gradients_received = False

			print("Master node is entering step: {}".format(i))
			self.async_bcast_step()
			self.async_bcast_layer_weights()
			
			# set the gradient fetch step and gather the request
			gradient_fetch_requests=self.async_fetch_gradient_start()

			# wait for enough gradients to be aggregated:
			while not enough_gradients_received:
				status = MPI.Status()
				MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)

				if status.tag-12 in self.grad_accumulator.model_index_range:
					layer_index = status.tag-12
					received_grad=self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
					
					# do gradient shape check here
					assert (received_grad.shape == self._model_shapes)

					# aggregate the gradient
					self.aggregate_gradient(gradient=received_grad, layer_idx=layer_index)

					self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1
				
				enough_gradients_received = True
				for j in self.grad_accumulator.gradient_aggregate_counter:
					enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)

			# average gradients and update the mode
			for i in range(len(self._grad_aggregate_buffer)):
				self._grad_aggregate_buffer[i] /= self._num_grad_to_collect

			# update the model on parameter sever using averaged model
			'''
			for layer_idx, layer in enumerate(self.module):
				if layer.is_fc_layer:
					fc_layer_idx = self.fc_layer_counts.index(layer_idx)
					update_params_dist_version(layer=layer, avg_grad=self._grad_aggregate_buffer[fc_layer_idx], learning_rate=self.lr)
			'''
			self.model_update()

			# reset essential elements
			self.meset_grad_buffer()
			self.grad_accumulator.meset_everything()
			self.cur_step += 1

	def init_model_shapes(self):
		for param_idx, param in enumerate(self.network.parameters()):
			self._model_shapes.append(param.size())

	def async_bcast_step(self):
		req_list = []
		for i in range(self.world_size):
			if i != 0:
				req_list.append(self.comm.isend(self.cur_step, dest=i, tag=10))
		for i in range(len(req_list)):
			req_list[i].wait()

	def async_bcast_layer_weights(self):
		request_layers = []
		for layer_idx, layer in enumerate(self.network.parameters()):
			request_workers = []
			layer_to_send = layer.data.numpy()
			for i in range(self.world_size):
				if i != 0:
					if layer.is_fc_layer:
						req = self.comm.Isend([layer_to_send, MPI.DOUBLE], dest=i, tag=11+layer_idx)
						request_workers.append(req)
			request_layers.append(request_workers)
		# TODO(hwang): check to see if these `wait` calls are necessary here
		for req_l in request_layers:
			for req_worker in req_l:
				req_worker.wait()

	def async_fetch_gradient_start(self):
		'''make gradient fetch requests and return the request list'''
		gradient_fetch_requests = [] # `graident_fetch_request` should have length of #fc_layer*num_grad_to_collect
		for layer_idx, layer in enumerate(self.network.parameters()):
			for k in range(self._num_grad_to_collect):
				req = self.comm.Irecv([self.grad_accumulator.gradient_aggregator[layer_idx][k], MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=88+layer_idx)
				gradient_fetch_requests.append(req)
		return gradient_fetch_requests

	def aggregate_gradient(self, gradient, layer_idx):
		'''keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`'''
		self._grad_aggregate_buffer[layer_idx] += gradient

	def model_update(self):
        """write model fetched from parameter server to local model"""
        new_state_dict = {}
        for param_idx,(key_name, param) in enumerate(self.network.state_dict().items()):
            assert param.size() == self._grad_aggregate_buffer[param_idx].shape
            tmp_dict = {key_name: self._grad_aggregate_buffer[param_idx].ToTensor()}
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

	def meset_grad_buffer(self):
		for i in range(len(self._grad_aggregate_buffer)):
			self._grad_aggregate_buffer[i] = np.zeros(self._grad_aggregate_buffer[i].shape)