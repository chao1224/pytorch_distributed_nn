from __future__ import print_function
import time
from random import shuffle
from copy import deepcopy

from mpi4py import MPI

from nn.nn import *

STEP_START_ = 1

MAX_NUM_ITERATIONS = 1000000

#MAX_NUM_ITERATIONS = 3

def update_params_dist_version(layer, avg_grad, learning_rate):
    '''update the network layer by layer'''
    shape_tmp = layer.get_shape
    layer.W -= learning_rate * avg_grad[0:shape_tmp[0], :]
    layer.b -= learning_rate * avg_grad[shape_tmp[0], :]

class GradientAccumulator(object):
	'''a simple class to implement gradient aggregator like the `Conditional Accumulators` in tensorflow'''
	def __init__(self, module, num_worker, fc_layer_num):
		# we will update this counter dynamically during the training process
		# the length of this counter should be number of fc layers in the network
		self.gradient_aggregate_counter = [0] * fc_layer_num

		# we used list to contain gradients of layers
		self.gradient_aggregator = []

		# TODO(hwang): we do not need to allocate so many space here since we need to aggregate the gradient
		# into each slot
		for layer_idx, layer in enumerate(module):
			if layer.is_fc_layer:
				tmp_aggregator = []
				for worker_idx in range(num_worker):
					tmp_aggregator.append(np.zeros((layer.get_shape[0]+1, layer.get_shape[1])))
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


class SyncReplicasMaster_NN(FC_NN):
	def __init__(self, comm, world_size, num_grad_to_collect, **kwargs):
		'''master node here, no rank needed since the rank will always be 0 for master node'''
		self.comm = comm   # get MPI communicator object
		self.world_size = world_size # total number of processes
		self.num_grad_to_collect = num_grad_to_collect # how many grads we want to collect in a certain iteration
		self.cur_step = STEP_START_
		self.fc_layer_counts = []
		self.lr = kwargs['learning_rate']

		self._num_grad_to_collect = self.world_size - 1
		# used to aggregate tmp gradients, the length is the same as # of fc layer 
		self._grad_aggregate_buffer = []
		# represent the `k` in our settings
		self._should_kill_threshold = 5

	def build_model(self, num_layers, layer_config):
		layers = []
		for i in range(num_layers):
			if layer_config['layer'+str(i)]['type'] == 'activation':
				if layer_config['layer'+str(i)]['name'] == 'sigmoid':
				    layers.append(SigmoidLayer())
				    layers[-1].layer_index = i
				elif layer_config['layer'+str(i)]['name'] == 'softmax':
				    layers.append(SoftmaxOutputLayer())
				    layers[-1].layer_index = i
			elif layer_config['layer'+str(i)]['type'] == 'fc':
				self.fc_layer_counts.append(i)
				layers.append(LinearLayer(layer_config['layer'+str(i)]['shape'][0], layer_config['layer'+str(i)]['shape'][1]))
				self._grad_aggregate_buffer.append(np.zeros(layers[-1].fetch_wrapped_shape))
				layers[-1].layer_index = i
		self.module = layers
		self.grad_accumulator = GradientAccumulator(module=self.module, num_worker=self.world_size-1, fc_layer_num=len(self.fc_layer_counts))
		return layers

	def train(self, training_set, validation_set):
		# the first step we need to do here is to sync fetch the inital worl_step from the parameter server
		# we still need to make sure the value we fetched from parameter server is 1
		# please note that step is start from one here
		self.async_bcast_step()

		# fake test here:
		for i in range(1, MAX_NUM_ITERATIONS):
			enough_gradients_received = False

			# workers should be killed at this iteration, this can change from iteration to iteration
			workers_should_kill = []
			# in current version we only use this for the 1st layer
			source_gathered = []

			print("Master node is entering step: {}".format(i))
			self.async_bcast_step()
			self.async_bcast_layer_weights()
			# set the gradient fetch step and gather the request
			gradient_fetch_requests=self.async_fetch_gradient_start()

			#remaining_req_indices = []
			received_req_indices = []
			# wait for enough gradients to be aggregated:
			while not enough_gradients_received:
				status = MPI.Status()
				req_index=MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)
				received_req_indices.append(req_index)

				# TODO(hwang): the layer indices parts are so hacky, definately re-arrange it sooner than later
				if status.tag-12 in self.fc_layer_counts:
					ori_layer_index = status.tag-12
				#if status.tag-TAG_LIST_[self.cur_step] in self.fc_layer_counts:
				#	ori_layer_index = status.tag-TAG_LIST_[self.cur_step]
					fc_index = self.fc_layer_counts.index(ori_layer_index)
					#received_grad=self.grad_accumulator.gradient_aggregator[fc_index][status.source-1]
					received_grad=self.grad_accumulator.gradient_aggregator[fc_index][0]
					
					# do gradient check here
					assert (received_grad.shape == self.module[ori_layer_index].fetch_wrapped_shape)

					# aggregate the gradient
					self.aggregate_gradient(gradient=received_grad, layer_idx=fc_index)

					if fc_index == 0:
						source_gathered.append(status.source)

					self.grad_accumulator.gradient_aggregate_counter[fc_index] += 1
					
					################################ straggler killing process ###############################################
#					if self.cur_step > 1000:
							
					if self.grad_accumulator.gradient_aggregate_counter[0] >= self._should_kill_threshold:
						#print("Start the killing process!")
						# start the killing process here:
						# generate the should kill worker list first:
						print("Start kill the worker")
						
						workers_should_kill = list(filter(lambda t: t not in source_gathered, [i for i in range(1, self.world_size)]))
						print("Should kill list", workers_should_kill)
						print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
						# this might be a naive solution, but for this version we wait the kill signal to be sent
						kill_req_list=self.send_kill_signal(should_kill_list=workers_should_kill)
						# try block sending here
						#self.send_kill_signal(should_kill_list=workers_should_kill)
					
						for req in kill_req_list:
							req.Wait()
						print("Sent the killing signal")
						break
					##########################################################################################################

				print(self.grad_accumulator.gradient_aggregate_counter)
				print('----------------------------------------------------------------------')
				
				enough_gradients_received = True
				for j_idx, j in enumerate(self.grad_accumulator.gradient_aggregate_counter):
					if j_idx == 0:
						enough_gradients_received = enough_gradients_received and (j >= self._should_kill_threshold)
					else:
						enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)

			# free all requests
			remaining_req_indices = [i for i in range(len(gradient_fetch_requests)) if i not in received_req_indices]

			#for req in reversed(gradient_fetch_requests):
			for i in remaining_req_indices:
				req = gradient_fetch_requests[i]
				# this is essential
				req.Cancel()

			# average gradients and update the mode
			for i in range(len(self._grad_aggregate_buffer)):
				self._grad_aggregate_buffer[i] /= self._num_grad_to_collect

			for layer_idx, layer in enumerate(self.module):
				if layer.is_fc_layer:
					fc_layer_idx = self.fc_layer_counts.index(layer_idx)
					update_params_dist_version(layer=layer, avg_grad=self._grad_aggregate_buffer[fc_layer_idx], learning_rate=self.lr)

			# reset essential elements
			self.meset_grad_buffer()
			self.grad_accumulator.meset_everything()
			self.cur_step += 1


	def async_bcast_step(self):
		req_list = []
		for i in range(self.world_size):
			if i != 0:
				req_list.append(self.comm.isend(self.cur_step, dest=i, tag=10))
		for i in range(len(req_list)):
			req_list[i].wait()

	def test_send(self):
		'''this is a test function to test the send and receive for a single layer'''
		buf = np.random.randn(20, 10)
		for i in range(self.world_size):
			if i != 0:
				self.comm.Isend([buf, MPI.DOUBLE], dest=i, tag=11)

	def async_bcast_layer_weights(self):
		request_layers = []
		for layer_idx, layer in enumerate(self.module):
			request_workers = []
			for i in range(self.world_size):
				if i != 0:
					if layer.is_fc_layer:
						req = self.comm.Isend([layer.fetch_wrapped_layer, MPI.DOUBLE], dest=i, tag=11+layer_idx)
						request_workers.append(req)
			request_layers.append(request_workers)
		# TODO(hwang): check to see if these `wait` calls are necessary here
		for req_l in request_layers:
			for req_worker in req_l:
				req_worker.wait()

	def async_fetch_gradient_start(self):
		'''make gradient fetch requests and return the request list'''
		gradient_fetch_requests = [] # `graident_fetch_request` should have length of #fc_layer*num_grad_to_collect
		#tmp_buffer = deepcopy(self.grad_accumulator.gradient_aggregator)
		for layer_idx, layer in enumerate(self.module):
			if layer.is_fc_layer:
				for k in range(self._num_grad_to_collect):
					i = self.fc_layer_counts.index(layer_idx)
					req = self.comm.Irecv([self.grad_accumulator.gradient_aggregator[i][0], MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=12+layer_idx)
					gradient_fetch_requests.append(req)
		return gradient_fetch_requests

	def aggregate_gradient(self, gradient, layer_idx):
		'''keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`'''
		self._grad_aggregate_buffer[layer_idx] += gradient

	def meset_grad_buffer(self):
		for i in range(len(self._grad_aggregate_buffer)):
			self._grad_aggregate_buffer[i] = np.zeros(self._grad_aggregate_buffer[i].shape)

	def send_kill_signal(self, should_kill_list):
		kill_req_list = []
		for worker_idx in should_kill_list:
			req=self.comm.isend(-1, dest=worker_idx, tag=77)
			kill_req_list.append(req)
		return kill_req_list