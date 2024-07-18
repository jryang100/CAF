import numpy as np


def conditional_errors(preds, labels, attrs):
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    cls_error = 1 - np.mean(preds == labels)
    idx = attrs == 0
    error_0 = 1 - np.mean(preds[idx] == labels[idx])
    error_1 = 1 - np.mean(preds[~idx] == labels[~idx])
    return cls_error, error_0, error_1
