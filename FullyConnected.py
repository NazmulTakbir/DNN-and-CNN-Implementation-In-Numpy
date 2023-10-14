import numpy as np

class FullyConnected:

    def __init__(self, no_inputs, no_outputs, optimizer='Adam', l2_lambda=0.1, dropout=0):
        self.no_inputs = no_inputs
        self.no_outputs = no_outputs
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda
        self.dropout = dropout
        self.init_weights()
        self.init_optimizer()

    def init_weights(self, init_type='xavier'):
        if init_type=='xavier':
            a = np.sqrt(6 / (self.no_inputs+self.no_outputs))
            self.weights = np.random.uniform(size=(self.no_outputs, self.no_inputs))
            self.weights = (self.weights * 2 * a) - a 
        elif init_type=='relu':
            self.weights = np.random.randn(size=(self.no_outputs, self.no_inputs))
            self.weights = self.weights * np.sqrt(2/self.no_inputs)
        else:
            raise Exception("Unknown Weight Initialization")

        self.biases = np.zeros((self.no_outputs, 1))

    def init_optimizer(self):
        if self.optimizer == 'SGD':
            pass

        elif self.optimizer == 'SGD-Momentum':
            self.beta1 = 0.9
            self.prev_weights_grad = None
            self.prev_biases_grad = None

        elif self.optimizer == 'Adam':
            self.avg_weight_grads = np.zeros((self.no_outputs, self.no_inputs))
            self.avg_bias_grads = np.zeros((self.no_outputs, 1))
            self.avg_squared_weight_grads = np.zeros((self.no_outputs, self.no_inputs))
            self.avg_squared_bias_grads = np.zeros((self.no_outputs, 1))

            self.beta1 = 0.9
            self.beta2 = 0.99
            self.itr = 1
        
        else:
            raise Exception("Unknown Optimizer")

    def validate_input_shape(self, input_shape):
        assert self.no_inputs == input_shape[0], f'Number of inputs ({input_shape[0]}) does not match number of inputs expected ({self.no_inputs})'
    
    def forward(self, input, is_training=True):
        self.validate_input_shape(input.shape) 
        
        output = np.matmul(self.weights, input) + self.biases

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

        # compute gradients
        weights_grad = np.matmul(grad_output, self.input.T) 
        biases_grad = np.sum(grad_output, axis=1, keepdims=True) 
        input_grad = np.matmul(self.weights.T, grad_output)

        weights_grad = weights_grad + self.l2_lambda * weights_grad
        biases_grad = biases_grad + self.l2_lambda * biases_grad 

        self.update(weights_grad, biases_grad, learning_rate)
        self.input = None

        return input_grad

    def update(self, weights_grad, biases_grad, learning_rate):

        if self.optimizer == 'SGD':
            pass

        elif self.optimizer == 'SGD-Momentum':
            if not self.prev_weights_grad is None:
                weights_grad = self.beta1 * self.prev_weights_grad + (1-self.beta1) * weights_grad
                biases_grad = self.beta1 * self.prev_biases_grad + (1-self.beta1) * biases_grad
            self.prev_weights_grad = weights_grad
            self.prev_biases_grad = biases_grad

        elif self.optimizer == 'Adam':
            self.avg_weight_grads = self.beta1 * self.avg_weight_grads + (1-self.beta1) * weights_grad
            self.avg_bias_grads = self.beta1 * self.avg_bias_grads + (1-self.beta1) * biases_grad

            self.avg_squared_weight_grads = self.beta2 * self.avg_squared_weight_grads + (1-self.beta2) * (weights_grad*weights_grad)
            self.avg_squared_bias_grads = self.beta2 * self.avg_squared_bias_grads + (1-self.beta2) * (biases_grad*biases_grad)

            weights_grad = self.avg_weight_grads / (1-self.beta1**self.itr)
            biases_grad = self.avg_bias_grads / (1-self.beta1**self.itr)

            squared_weight_grads = self.avg_squared_weight_grads / (1-self.beta2**self.itr)
            squared_bias_grads = self.avg_squared_bias_grads / (1-self.beta2**self.itr)

            weights_grad = weights_grad / (np.sqrt(squared_weight_grads)+1e-7)
            biases_grad = biases_grad / (np.sqrt(squared_bias_grads)+1e-7)

            self.itr += 1
        
        else:
            raise Exception("Unknown Optimizer")

        self.weights -= 1/self.input.shape[0] * learning_rate * weights_grad 
        self.biases -= 1/self.input.shape[0] * learning_rate * biases_grad

    def get_n_params(self):
        return self.no_inputs * self.no_outputs + self.no_outputs
    
