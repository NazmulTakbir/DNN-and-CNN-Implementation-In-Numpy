from FullyConnected import FullyConnected
from ReLU import ReLU
from CrossEntropyLoss import CrossEntropyLoss
import numpy as np
from Convolution import Convolution
from Flatten import Flatten
from MaxPool import MaxPool

class Model:

    def __init__(self): 
        # LeNet-5
        self.layers = [
            Convolution(1, 6, 5, 1, 0),
            ReLU(),
            MaxPool(6, 2, 2, 0),
            Convolution(6, 16, 5, 1, 0),
            ReLU(),
            MaxPool(16, 2, 2, 0),
            Convolution(16, 120, 5, 1, 0),
            ReLU(),
            Flatten(),
            FullyConnected(120, 84),
            ReLU(),
            FullyConnected(84, 10),
        ]
        self.loss = CrossEntropyLoss(10)

        print("Model has {} parameters".format(self.get_n_params()))
        
        self.learning_rate = 0.1

    def forward(self, input, target=None, is_training=True):
        for layer in self.layers:
            input = layer.forward(input, is_training=is_training)

        if target is None:
            probs = self.loss.softmax(input)
            preds = np.argmax(probs, axis=0)
            return preds.reshape(-1)
        else:
            probs, avg_loss, total_loss = self.loss.forward(input, target, is_training=is_training)
            preds = np.argmax(probs, axis=0)
            return preds.reshape(-1), avg_loss, total_loss
    
    def backward(self, target):
        grad_output = self.loss.backward(target)
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, self.learning_rate)
    
    def get_n_params(self):
        n_params = 0
        for layer in self.layers:
            n_params += layer.get_n_params()
        return n_params