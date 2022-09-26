import numpy as np
import mlflow

def get_or_create_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        return mlflow.get_experiment_by_name(experiment_name)

def get_experiment_id(experiment_name):
    return get_or_create_experiment(experiment_name).experiment_id

