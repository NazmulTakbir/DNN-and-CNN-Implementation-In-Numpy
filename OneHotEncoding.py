import numpy as np

class OneHotEncoding:

    def __init__(self, classes):
        self.classes = np.unique(classes)
        self.no_classes = len(self.classes)
        self.cls_to_idx = {c.item(): i for i, c in enumerate(self.classes)}
        self.idx_to_cls = {i: c.item() for i, c in enumerate(self.classes)}

    def transform(self, y):
        one_hot_encoding = np.zeros((self.no_classes, len(y)))
        for i, c in enumerate(y):
            one_hot_encoding[self.cls_to_idx[c.item()], i] = 1
        return one_hot_encoding
    
    def index_to_class(self, idx_list):
        return np.array([self.idx_to_cls[idx.item()] for idx in idx_list])
    
    def inverse_transform(self, encoding):
        return self.index_to_class(np.argmax(encoding, axis=0))