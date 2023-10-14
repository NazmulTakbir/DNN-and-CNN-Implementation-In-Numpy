import numpy as np

class ReLU:
    
    def forward(self, input, is_training=True):
        if is_training:
            self.input = input
        return np.maximum(input, 0)

    def backward(self, grad_output, learning_rate):
        input_grad = (self.input > 0) * grad_output
        self.input = None
        return input_grad  
    
    def get_n_params(self):
        return 0