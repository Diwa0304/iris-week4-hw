import mlflow
def setup_mlflow(experiment_name, host_address = "http://121.0.0.1:8100"):
    mlflow.set_tracking_uri(host_address)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking set to: {mlflow.get_tracking_uri()}\nExperiment name: {experiment_name}")
    print("Experiment set successfully")