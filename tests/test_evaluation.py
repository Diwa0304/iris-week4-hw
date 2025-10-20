import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load("models/iris_classifier.joblib")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc > 0.7, f"Model accuracy too low: {acc}"
