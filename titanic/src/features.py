import pandas as pd


def get_features(df):
    features=[
        "Pclass","Age","Fare","sex_bin",
        "Embarked_C","Embarked_Q","Embarked_S",
        "FamilySize","IsAlone","IsChild"
    ]
    return df[features]
