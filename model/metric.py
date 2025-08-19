import torch
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def Auc(output, target):
    with torch.no_grad():
        auc_score = roc_auc_score(y_true=target,y_score=output)
    return auc_score

def Accuracy(output, target): 
    with torch.no_grad():
        acc_sc = accuracy_score(y_true = target, y_pred = output)
    return acc_sc

def MixedMetrics_DEPRECATED(output, target):
    with torch.no_grad():
        tn, fp, fn, tp = confusion_matrix(y_true = target, y_pred = output, labels=[0,1]).ravel()
        fpr = fp / ( fp + tn )
        f1score = 2 * tp / (2 * tp + fp + fn)
        precision = tp / ( tp + fp )
        sensitivity = tp / ( tp + fn )
        specificity = tn / ( tn + fp )
    return [('fpr', fpr), ('f1score', f1score), ('precision', precision), ('sensitivity', sensitivity), ('specificity', specificity)]

class BaseMetrics():
    def __call__(self, output, target):
        with torch.no_grad():
            if len(np.unique(target, return_counts=False)) == 1:
                auc_score = float('nan')
            else:
                auc_score = roc_auc_score(y_true=target,y_score=output)
            acc_sc = accuracy_score(y_true = target, y_pred = np.array(output) > 0.5)
        return [('AUC', auc_score), ('Accuracy', acc_sc)]

    def get_keys(self):
        return ['AUC', 'Accuracy']

class MixedMetrics():
    
    def __call__(self, output, target):
        with torch.no_grad():
            tn, fp, fn, tp = confusion_matrix(y_true = target, y_pred = np.array(output)  > 0.5, labels=[0,1]).ravel()
            fpr = fp / ( fp + tn )
            f1score = 2 * tp / (2 * tp + fp + fn)
            precision = tp / ( tp + fp )
            sensitivity = tp / ( tp + fn )
            specificity = tn / ( tn + fp )
        return [('fpr', fpr), ('f1score', f1score), ('precision', precision), ('sensitivity', sensitivity), ('specificity', specificity)]
    
    def get_keys(self):
        return ['fpr', 'f1score', 'precision', 'sensitivity', 'specificity']




