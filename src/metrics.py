"""Task 4.2: the three metric functions, written by hand."""
import numpy as np

def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

def confusion_matrix(y_true, y_pred, num_classes=6):
    """Entry [j, k] counts windows of true activity j that were predicted as activity k."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return matrix

def per_class_accuracy(y_true, y_pred, num_classes=6):
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    return np.diag(matrix) / matrix.sum(axis=1)
