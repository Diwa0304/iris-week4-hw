import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import joblib
import os
import json
from mlflow import MlflowClient
import mlflow.pyfunc 
from mlflow.exceptions import MlflowException

# change what model you want to use (registered model name)
MODEL_NAME = "IRIS-Classifier-LogReg"

TRACKING_URI = "http://127.0.0.1:8100"
mlflow.set_tracking_uri(TRACKING_URI)

def fetch_and_load_latest_model(model_name: str):
    """
    Fetches the latest model version from the MLflow Model Registry and loads it.
    """
    model_uri = f"models:/{model_name}/latest"
    print(f"Attempting to load latest model from URI: {model_uri}")
    
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded the latest model for '{model_name}'.")

        return loaded_model

    except MlflowException as e:
        print(f"Error loading model from registry: {e}")
        print("Ensure the model name is correct, the MLflow Tracking Server is running, and the model is registered.")
        return None
      
def fetch_and_load_best_model(model_name: str, metric_key: str = "accuracy"):
    """
    Fetches the model version with the highest value for a specific metric.
    """
    client = MlflowClient()
    
    filter_string = f"name='{model_name}'"
    ordered_versions = client.search_model_versions(
        filter_string=filter_string,
        order_by=[f"metrics.{metric_key} DESC"],
        max_results=1
    )
    
    if not ordered_versions:
        print(f"No model versions found for registered model: {model_name}")
        return None
        
    best_version_info = ordered_versions[0]
    best_version = best_version_info.version
    
    print(f"Best model version found (by {metric_key}): Version {best_version} with {metric_key}={best_version_info.get_metric(metric_key)}")

    # Load the models
    model_uri = f"models:/{model_name}/{best_version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    return loaded_model

def run_evaluation(model_type:str):
    """
    The main evaluation pipeline function.
    """
    if model_type=="latest":
        model = fetch_and_load_latest_model(MODEL_NAME)
    elif model_type=="best":
        model = fetch_and_load_best_model(MODEL_NAME)
    else:
        print("model type can only be 'best' or 'latest'"

    if model is None:
        print("Evaluation pipeline aborted due to failure to load model.")
        return
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
              
run_evaluation("latest")