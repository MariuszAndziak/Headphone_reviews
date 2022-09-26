import numpy as np
import mlflow

def get_or_create_experiment(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        return mlflow.get_experiment_by_name(experiment_name)


def get_experiment_id(experiment_name: str):
    return get_or_create_experiment(experiment_name).experiment_id


def mlflow_start(df, model_obj, feats, target, exepriment_id, run_name = None):

    model_name, model = model_obj

    if not run_name:
        run_name = model_name
    
    with mlflow.start_run(experiment_id=get_experiment_id(exepriment_id), run_name = run_name) as run:
        pass