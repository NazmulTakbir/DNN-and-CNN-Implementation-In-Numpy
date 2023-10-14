import numpy as np

class CrossEntropyLoss:

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def softmax(self, input):
        assert input.shape[0] == self.num_classes

        max_input = np.ones(input.shape) * 50
        min_input = np.ones(input.shape) * -50

        input = np.minimum(input, max_input)
        input = np.maximum(input, min_input)

        exps = np.exp(input)
        return exps / np.sum(exps, axis=0)
    
    def forward(self, input, target, is_training=True):
        assert input.shape[0] == self.num_classes and target.shape[0] == self.num_classes
        
        probs = self.softmax(input)
        log_probs = np.log(probs)
        total_loss = -np.sum(log_probs * target)
        avg_loss = total_loss / input.shape[1]

        if is_training:
            self.input = input

        return probs, avg_loss, total_loss

    def backward(self, target):
        probs = self.softmax(self.input)
        input_grad = (probs-1)*target+probs*(1-target)
        self.input = None
        return input_grad