import numpy as np

class Convolution:

    def __init__(self, no_channels_in, no_filters, filter_size, stride, padding, optimizer='Adam', l2_lambda=0.1, dropout=0):
        self.no_channels_in = no_channels_in
        self.no_filters = no_filters
        self.filter_size = filter_size
        self.stride = max(1, int(stride))
        self.padding = padding
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda
        self.dropout = dropout
        self.init_weights()
        self.init_optimizer()

    def init_weights(self, init_type='xavier'):
        size = (self.no_filters, self.no_channels_in, self.filter_size, self.filter_size)
        if init_type=='xavier':
            a = np.sqrt(6 / (self.no_channels_in+self.no_filters))
            self.filters = np.random.uniform(size=size) 
            self.filters = (self.filters * 2 * a) - a
        elif init_type=='relu':
            self.filters = np.random.randn(size=size) 
            self.filters = self.filters * np.sqrt(2/self.no_inputs)
        else:
            raise Exception("Unknown Weight Initialization")

        
        self.biases = np.zeros(self.no_filters) 

    def init_optimizer(self):
        if self.optimizer == 'SGD':
            pass

        elif self.optimizer == 'SGD-Momentum':
            self.beta1 = 0.9
            self.prev_grad_filters = None
            self.prev_grad_biases = None

        elif self.optimizer == 'Adam':
            size = (self.no_filters, self.no_channels_in, self.filter_size, self.filter_size)
            self.avg_filter_grads = np.zeros((size))
            self.avg_bias_grads = np.zeros(self.no_filters)
            self.avg_squared_filter_grads = np.zeros((size))
            self.avg_squared_bias_grads = np.zeros(self.no_filters)
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.itr = 1
        
        else:
            raise Exception("Unknown Optimizer")

    def set_input_shape(self, d_no, ch_no, h_in, w_in):
        self.d_no, self.ch_no, self.h_in, self.w_in = d_no, ch_no, h_in, w_in

        self.h_out = int((h_in + 2 * self.padding - self.filter_size) / self.stride + 1)
        self.w_out = int((w_in + 2 * self.padding - self.filter_size) / self.stride + 1)

        if self.stride > 1:
            h_max = int((h_in + 2 * self.padding - self.filter_size)+ 1)
            w_max = int((w_in + 2 * self.padding - self.filter_size) + 1)
            self.expand_col = np.zeros((d_no, self.no_filters, self.h_out, w_max))
            r = c = 0
            for _ in range(self.w_out):
                self.expand_col[:, :, r, c] = 1
                c += self.stride
                r += 1

            self.expand_row = np.zeros((d_no, self.no_filters, h_max, self.w_out))
            r = c = 0
            for _ in range(self.h_out):
                self.expand_row[:, :, r, c] = 1
                c += 1
                r += self.stride

    def validate_input_shape(self, input_shape):
        try:
            if self.d_no != input_shape[0]:
                self.set_input_shape(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
            assert self.ch_no == input_shape[1], f'Number of channels in input ({input_shape[1]}) does not match number of channels expected ({self.ch_no})'
            assert self.h_in == input_shape[2], f'Height of input ({input_shape[2]}) does not match height expected ({self.h_in})'
            assert self.w_in == input_shape[3], f'Width of input ({input_shape[3]}) does not match width expected ({self.w_in})'
        except:
            self.set_input_shape(input_shape[0], input_shape[1], input_shape[2], input_shape[3])

    def forward(self, input, is_training=True):
        self.validate_input_shape(input.shape)

        if self.padding > 0:
            input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        in_strides = input.strides
        view_shape = (self.d_no, self.h_out, self.w_out, self.no_channels_in, self.filter_size, self.filter_size)
        view_strides = (in_strides[0], self.stride*in_strides[2], self.stride*in_strides[3], in_strides[1], in_strides[2], in_strides[3])
        view_slices = np.lib.stride_tricks.as_strided(input, shape=view_shape, strides=view_strides)

        output = np.einsum('abcxyz,mxyz->ambc', view_slices, self.filters) + self.biases.reshape(1, self.no_filters, 1, 1)

        if is_training:
            self.input = input
            if self.dropout > 0:
                self.dropout_mask = (np.random.uniform(size=output.shape) > self.dropout).astype(np.uint8)
                output = (output * self.dropout_mask) / (1-self.dropout)

        return output 

    def backward(self, grad_output, learning_rate):
        # handle dropout
        if self.dropout > 0:
            grad_output = (grad_output * self.dropout_mask) / (1-self.dropout)

        input = self.input

        # pad input if needed
        if self.padding > 0:
            input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        
        # expand grad_output if needed - insert column of 0s and row of 0s for each stride
        if self.stride > 1:
            grad_output = np.matmul(self.expand_row, np.matmul(grad_output, self.expand_col))
        
        # compute gradient of filters
        in_strides = input.strides
        view_shape = (input.shape[0], input.shape[1], self.filter_size, self.filter_size, grad_output.shape[2], grad_output.shape[3])
        view_strides = (in_strides[0], in_strides[1], in_strides[2], in_strides[3], in_strides[2], in_strides[3])
        view_slices = np.lib.stride_tricks.as_strided(input, shape=view_shape, strides=view_strides)
        grad_filters = np.einsum('abcdef,axef->xbcd', view_slices, grad_output)

        # compute gradient of biases
        grad_biases = np.sum(grad_output, axis=(0, 2, 3))
        
        # compute gradient of input
        rotated_filters = np.rot90(self.filters, 2, axes=(2, 3))
        grad_output_padded = np.pad(grad_output, ((0, 0), (0, 0), (self.filter_size - 1, self.filter_size - 1), (self.filter_size - 1, self.filter_size - 1)), 'constant')
        
        in_strides = grad_output_padded.strides
        view_shape = (input.shape[0], input.shape[2], input.shape[3], self.no_filters, self.filter_size, self.filter_size)
        view_strides = (in_strides[0], in_strides[2], in_strides[3], in_strides[1], in_strides[2], in_strides[3])
        view_slices = np.lib.stride_tricks.as_strided(grad_output_padded, shape=view_shape, strides=view_strides)
        grad_input = np.einsum('abcdef,dxef->axbc', view_slices, rotated_filters)
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        grad_filters = grad_filters + self.l2_lambda * grad_filters
        grad_biases = grad_biases + self.l2_lambda * grad_biases 

        self.update(grad_filters, grad_biases, learning_rate)
        self.input = None

        return grad_input

    def update(self, grad_filters, grad_biases, learning_rate):

        if self.optimizer == 'SGD':
            pass

        elif self.optimizer == 'SGD-Momentum':
            if self.prev_grad_filters is not None:
                grad_filters = self.beta1 * self.prev_grad_filters + (1 - self.beta1) * grad_filters
                grad_biases = self.beta1 * self.prev_grad_biases + (1 - self.beta1) * grad_biases
            self.prev_grad_filters = grad_filters
            self.prev_grad_biases = grad_biases

        elif self.optimizer == 'Adam':
            self.avg_filter_grads = self.beta1 * self.avg_filter_grads + (1-self.beta1) * grad_filters
            self.avg_bias_grads = self.beta1 * self.avg_bias_grads + (1-self.beta1) * grad_biases

            self.avg_squared_filter_grads = self.beta2 * self.avg_squared_filter_grads + (1-self.beta2) * (grad_filters*grad_filters)
            self.avg_squared_bias_grads = self.beta2 * self.avg_squared_bias_grads + (1-self.beta2) * (grad_biases*grad_biases)

            grad_filters = self.avg_filter_grads / (1-self.beta1**self.itr)
            grad_biases = self.avg_bias_grads / (1-self.beta1**self.itr)

            squared_filter_grads = self.avg_squared_filter_grads / (1-self.beta2**self.itr)
            squared_bias_grads = self.avg_squared_bias_grads / (1-self.beta2**self.itr)

            grad_filters = grad_filters / (np.sqrt(squared_filter_grads)+1e-7)
            grad_biases = grad_biases / (np.sqrt(squared_bias_grads)+1e-7)

            self.itr += 1
        
        else:
            raise Exception("Unknown Optimizer")

        self.filters -= 1/self.d_no * learning_rate * grad_filters
        self.biases -= 1/self.d_no * learning_rate * grad_biases
    
    def get_n_params(self):
        return self.filters.size + self.biases.size