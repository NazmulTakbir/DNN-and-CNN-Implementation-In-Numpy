from Model import Model
from OneHotEncoding import OneHotEncoding
import time
from Numta_Loader import Numta_Loader
from Metrics import get_metrics, log_metrics, save_metrics
import pickle

model = Model()
img_size = 32 

train_loader = Numta_Loader(['Dataset/train'], img_size, in_memory=True)
val_loader = Numta_Loader(['Dataset/val'], img_size)

one_hot_encoder = OneHotEncoding([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

start_time = time.time()
print("Starting training...")
print('-------------------------------------------------------------------------------------------')

n_epochs = 20
train_batch_size = 512

metrics = []
for epoch in range(1, n_epochs+1):

    no_mini_batches = train_loader.get_size()//train_batch_size
    for i in range(no_mini_batches):
        res = train_loader.get_batch(train_batch_size)
        data, labels = res['data'], res['labels']
        labels = one_hot_encoder.transform(labels)
        model.forward(data, labels)
        model.backward(labels)

        duration = str(int(time.time() - start_time)) + 's'
        print(f'Epoch: {epoch}, Mini-batch: {i+1}/{no_mini_batches}, Duration: {duration}', end='\r')

    train_loss, train_accuracy, train_f1, _, _ = get_metrics(model, one_hot_encoder, train_loader)
    val_loss, val_accuracy, val_f1, _, _ = get_metrics(model, one_hot_encoder, val_loader)

    log_metrics(start_time, epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1)
    metrics.append([epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1])

print('Training complete!')
save_metrics(metrics, 'metrics.csv')

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved!')
