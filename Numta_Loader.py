import os
import numpy as np
import random
import cv2

class Numta_Loader:

    def __init__(self, root_dirs, img_size, preprocessor=None, has_labels=True, in_memory=False):
        self.img_size = img_size
        self.preprocessor = preprocessor
        self.has_labels = has_labels
        self.in_memory = in_memory
        self.meta_data = []
        if has_labels:
            for root_dir in root_dirs:
                meta_data = open(os.path.join(root_dir, 'labels.csv')).readlines()[1:]
                meta_data = [label.strip().split(",") for label in meta_data]
                meta_data = [[os.path.join(root_dir, label[0]), label[1]] for label in meta_data]
                self.meta_data += meta_data
        else:
            allowed_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'PNG', 'JPG', 'JPEG', 'BMP']
            for root_dir in root_dirs:
                meta_data = os.listdir(root_dir)
                meta_data = [[os.path.join(root_dir, img_name), -1] for img_name in meta_data if img_name.split(".")[-1] in allowed_extensions]
                self.meta_data += meta_data

        if in_memory:
            print("Loading images into memory...")
            self.images = {}
            for data in self.meta_data:
                self.images[data[0]] = self.load_image(data[0])
            print("Images loaded into memory!")

    def get_size(self):
        return len(self.meta_data)

    def load_image(self, img_path):
        if self.preprocessor is not None:
            img = self.preprocessor(img_path, self.img_size)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img.shape != (self.img_size, self.img_size):
                img = cv2.resize(img, (self.img_size, self.img_size))
            img = np.array(img)
        return img.astype(np.uint8)

    def get_batch(self, batch_size, shuffle=True, offset=0):
        if shuffle:
            random.shuffle(self.meta_data)
            batch = self.meta_data[:batch_size]
        else:
            batch = self.meta_data[offset:offset+batch_size]

        batch_data = np.zeros((batch_size, 1, self.img_size, self.img_size))
        batch_label = np.zeros(batch_size)
        batch_filenames = []

        for i, data in enumerate(batch):
            if self.in_memory:
                batch_data[i, :, :, :] = self.images[data[0]]
            else:
                batch_data[i, :, :, :] = self.load_image(os.path.join(data[0]))
            batch_label[i] = int(data[1])
            batch_filenames.append(data[0].split("/")[-1])
        
        return {
            'data': batch_data/255.0,
            'labels': batch_label.astype(np.uint8),
            'filenames': batch_filenames
        }