import numpy as np

class Flatten:

    def forward(self, input, is_training=True):
        self.input_shape = input.shape
        input = input.reshape(input.shape[0], -1)
        input = np.transpose(input, axes=[1,0])
        return input

    def backward(self, grad_output, learning_rate):
        return grad_output.T.reshape(self.input_shape)
    
    def get_n_params(self):
        return 0