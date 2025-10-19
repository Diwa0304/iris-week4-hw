from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os
import pandas as pd

df = pd.read_csv("data/iris_1.csv")

encoder = OrdinalEncoder() 
df["target"] = encoder.fit_transform(df[["species"]]).astype(int) 
X = df.drop(columns=["species", "target"]) 
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/iris_classifier.joblib")
joblib.dump(encoder, "models/encoder.joblib")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

