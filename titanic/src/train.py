import pandas as pd
import sklearn

import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parents[0]
sys.path.append(PROJECT_ROOT)

from data_loader import load_data
from preprocessing import compute_missing_params, preprocess_data
from features import get_features
from models import train_lr, train_rf
from submit import make_submission

def main():
    train_data = load_data('./data/raw/train.csv')
    test_data = load_data('./data/raw/test.csv')

    params = compute_missing_params(train_data)
    train_clean, test_clean = preprocess_data(train_data, test_data, params)
    X_train = get_features(train_clean)
    X_test  = get_features(test_clean)
    passenger_ids = test_clean["PassengerId"]
    y_train = train_clean["Survived"].astype(int)

    print(X_train.shape)
    print(X_test.shape)
    print(X_train.isnull().sum().sum())

    rf_model = train_rf(X_train, y_train, n_estimators=300, min_samples_leaf=5)

    make_submission(rf_model, X_test, passenger_ids, "./submissions/submission.csv")

if __name__ == "__main__":
    main()
