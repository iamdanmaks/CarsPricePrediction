import joblib
import os
import pandas as pd

from hashlib import sha256
from .preprocess import preprocess
from lightgbm import LGBMRegressor


def check_version():
    with open(os.path.dirname(__file__) + '/utils/checksum.sha256', "r") as f:
        checksum = f.read()
    
    with open(os.path.dirname(__file__) + '/utils/train.csv', "rb") as f:
        bts = f.read()
        actual_checksum = sha256(bts).hexdigest()

    if checksum != actual_checksum:
        train(builtin=True)
        with open(os.path.dirname(__file__) + '/utils/train.csv', "rb") as f:
            bts = f.read()
            with open(os.path.dirname(__file__) + '/utils/checksum.sha256', "w") as fl:
                fl.write(sha256(bts).hexdigest())


def train(train_set="default", builtin=False):
    if train_set == "default":
        train_set = os.path.dirname(__file__) + '/utils/train.csv'

    if not isinstance(train_set, pd.DataFrame):
        df = pd.read_csv(train_set).drop(columns=["Unnamed: 0"])
    else:
        df = train_set
    # validate schema and dtypes
    X_train, y_train = preprocess(df, train=True)
    
    model = LGBMRegressor(
        random_state=42,
        objective='mape',
        num_leaves=100,
        feature_fraction=0.9,
        max_depth=-1,
        learning_rate=0.03,
        num_iterations=1300,
        subsample=0.5
    )
    model.fit(X_train, y_train)

    if builtin:
        joblib.dump(model, os.path.dirname(__file__) + '/utils/train.csv')

    return model


def save_model(model, filename='model'):
    joblib.dump(model, f'./{filename}.pkl')
