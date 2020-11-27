from .predict import predict

import numpy as np


def score_model(model, X_test, y_test, eval_func):
    return eval_func(y_test, predict(model, X_test))


def score_model_pred(y_true, y_pred, eval_func):
    return eval_func(y_true, y_pred)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
