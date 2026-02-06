import pandas as pd


def get_features(df):
    features=[
        "Pclass","Age","Fare","sex_bin",
        "Embarked_C","Embarked_Q","Embarked_S",
        "FamilySize","IsAlone","IsChild","Title_Master", "Title_Miss","Title_Mrs", "Title_Mr", "Title_Rare",
    ]
    return df[features]
