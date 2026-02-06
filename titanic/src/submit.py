import pandas as pd

def make_submission(model, X_test, passenger_ids, filename):
    preds = model.predict(X_test)
    sub   = pd.DataFrame({"PassengerId": passenger_ids, "Survived":preds.astype(int)})
    sub.to_csv(filename, index=False)
    return preds
