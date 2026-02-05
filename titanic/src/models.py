import pandas as pd

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF

def train_lr(X, y, C=1.0):
    model = LR(max_iter=500, C=C)
    model.fit(X, y)
    return model

def train_rf(X, y, **kwargs):
    model = RF(random_state=42, **kwargs)
    model.fit(X, y)
    return model
