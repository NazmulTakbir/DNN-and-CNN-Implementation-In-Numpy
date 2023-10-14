import numpy as np
import warnings
import time

def macro_f1(preds, labels):
    tp = np.sum(preds * labels, axis=1)
    fp = np.sum(preds * (1 - labels), axis=1)
    fn = np.sum((1 - preds) * labels, axis=1)

    predicted_positives = tp+fp
    actual_positives = tp+fn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = tp / predicted_positives
        recall = tp / actual_positives
        precision[np.isnan(precision)] = 0
        recall[np.isnan(recall)] = 0

        f1 = 2 * precision * recall / (precision + recall)
        f1[np.isnan(f1)] = 0

    macro_f1 = np.mean(f1).item()
    return macro_f1

def get_metrics(model, one_hot_encoder, data_loader):
    offset = count = correct = loss = 0
    batch_size = min(512, data_loader.get_size())
    all_preds = all_labels = all_filenames = None
    
    while offset < data_loader.get_size():
        res = data_loader.get_batch(batch_size, offset=offset, shuffle=False)
        data, labels, filenames = res['data'], res['labels'], res['filenames']
        preds, _, total_loss = model.forward(data, one_hot_encoder.transform(labels), is_training=False)
        loss += total_loss
        count += data.shape[0]
        correct += np.sum(preds == labels).item()
        offset += batch_size
        batch_size = min(batch_size, data_loader.get_size() - offset)

        if all_preds is None:
            all_preds = preds
            all_labels = labels
            all_filenames = filenames
        else:
            all_preds = np.concatenate((all_preds, preds))
            all_labels = np.concatenate((all_labels, labels))
            all_filenames += filenames

        print("Completed: {}/{}".format(offset, data_loader.get_size()), end="\r")

    accuracy = correct / count
    loss = (loss / count).item()
    f1 = macro_f1(one_hot_encoder.transform(all_preds), one_hot_encoder.transform(all_labels))

    return loss, accuracy, f1, all_preds, all_filenames

def get_predictions(model, data_loader):
    offset = 0
    batch_size = min(512, data_loader.get_size())
    all_preds = all_filenames = None
    
    while offset < data_loader.get_size():
        res = data_loader.get_batch(batch_size, offset=offset, shuffle=False)
        data, _, filenames = res['data'], res['labels'], res['filenames']
        preds = model.forward(data, is_training=False)
        offset += batch_size
        batch_size = min(batch_size, data_loader.get_size() - offset)

        if all_preds is None:
            all_preds = preds
            all_filenames = filenames
        else:
            all_preds = np.concatenate((all_preds, preds))
            all_filenames += filenames

    return all_preds, all_filenames

def log_metrics(start_time, epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1):
    duration = str(int(time.time() - start_time)) + 's'
    print(f'Epoch: {epoch}, Duration: {duration}')
    print('Type'.ljust(10), 'Loss'.ljust(10), 'Accuracy'.ljust(10), 'F1'.ljust(10))
    print('Train'.ljust(10), str(round(train_loss, 2)).ljust(10), str(round(train_accuracy, 2)).ljust(10), str(round(train_f1, 2)).ljust(10))
    print('Val'.ljust(10), str(round(val_loss, 2)).ljust(10), str(round(val_accuracy, 2)).ljust(10), str(round(val_f1, 2)).ljust(10))
    print('-------------------------------------------------------------------------------------------')

def save_metrics(metrics, file_path):
    with open(file_path, 'w') as f:
        f.write('Epoch,Train_Loss,Train_Accuracy,Train_F1,Val_Loss,Val_Accuracy,Val_F1\n')
        for metric in metrics:
            f.write(','.join([str(m) for m in metric]) + '\n')
    print('Metrics saved to', file_path)