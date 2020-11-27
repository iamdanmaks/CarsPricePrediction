from .preprocess import preprocess

import joblib
import os

import numpy as np
import pandas as pd


def predict(model, pred_set):
    if not isinstance(pred_set, pd.DataFrame):
        df = pd.read_csv(pred_set)
    else:
        df = pred_set
    # validate schema and dtypes
    X_pred = preprocess(df.drop(columns=['Unnamed: 0']), train=False)

    return np.array([np.exp(p) for p in model.predict(X_pred)])


def load_model(path='default'):
    if path == 'default':
        return joblib.load(os.path.dirname(__file__) + '/utils/model.pkl')
    return joblib.load(path)
