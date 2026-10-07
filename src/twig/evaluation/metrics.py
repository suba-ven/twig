import numpy as np


def rmse(prediction, target, axis=None):
    return np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2, axis=axis))


def mae(prediction, target, axis=None):
    return np.mean(np.abs(np.asarray(prediction) - np.asarray(target)), axis=axis)
