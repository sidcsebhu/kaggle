import pandas as pd

def fill_age(row, age_median):
    if pd.isnull(row["Age"]):
        return age_median[row["Sex"], row["Pclass"]]
    else:
        return row["Age"]


def compute_missing_params(train):
    params = {}
    age_imputed          = train.groupby(["Sex", "Pclass"])
    params["age_median"] = age_imputed["Age"].median()
    params["Embarked"]   = train["Embarked"].mode()[0]
    params["Fare"]       = train["Fare"].median()

    return params


def preprocess_data(train, test, params):
    full = pd.concat([train, test], ignore_index=True)
    
    full["sex_bin"]  = full["Sex"].map({"male":0,"female":1})
    full["Age"]      = full.apply(fill_age, args=(params["age_median"],), axis=1)
    
    full["Embarked"].fillna(params["Embarked"], inplace=True)
    full["Fare"].fillna(params["Fare"], inplace=True)
    
    embark_series    = pd.get_dummies(full["Embarked"], prefix="Embarked")
    full             = pd.concat([full, embark_series], axis=1)
    full["FamilySize"] = full["SibSp"]+full["Parch"]+1
    full["IsAlone"]   = (full["FamilySize"] == 1).astype(int)
    full["IsChild"]   = (full["Age"] < 16).astype(int)
    
    full.drop(columns=["Name", "Sex", "Ticket", "Cabin", "SibSp", "Parch","Embarked"], inplace=True)
    
    train_clean      = full.iloc[:len(train)]
    test_clean       = full.iloc[len(train):]
    
    return train_clean, test_clean
