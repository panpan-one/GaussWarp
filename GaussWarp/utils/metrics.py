import numpy as np


def MAE(pred, true, mask=None):
    if mask is None:
        return np.mean(np.abs(pred - true))
    residual = (pred - true) * mask
    return np.sum(np.abs(residual)) / max(np.sum(mask), 1)


def MSE(pred, true, mask=None):
    if mask is None:
        return np.mean((pred - true) ** 2)
    residual = (pred - true) * mask
    return np.sum(residual ** 2) / max(np.sum(mask), 1)


def metric(pred, y, y_mask=None, **kwargs):
    return {
        "MAE": MAE(pred, y, y_mask),
        "MSE": MSE(pred, y, y_mask),
    }
