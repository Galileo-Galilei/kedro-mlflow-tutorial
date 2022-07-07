"""Project pipelines."""
from platform import python_version
from typing import Dict

from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory

from kedro_mlflow_tutorial import __version__ as PROJECT_VERSION
from kedro_mlflow_tutorial.pipelines.etl_app.pipeline import create_etl_pipeline
from kedro_mlflow_tutorial.pipelines.ml_app.pipeline import create_ml_pipeline
from kedro_mlflow_tutorial.pipelines.user_app.pipeline import create_user_app_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    etl_pipeline = create_etl_pipeline()
    etl_instances_pipeline = etl_pipeline.only_nodes_with_tags("etl_instances")
    etl_labels_pipeline = etl_pipeline.only_nodes_with_tags("etl_labels")

    ml_pipeline = create_ml_pipeline()
    inference_pipeline = ml_pipeline.only_nodes_with_tags("inference")
    training_pipeline_ml = pipeline_ml_factory(
        training=ml_pipeline.only_nodes_with_tags("training"),
        inference=inference_pipeline,
        input_name="instances",
        log_model_kwargs=dict(
            artifact_path="kedro_mlflow_tutorial",
            conda_env={
                "python": python_version(),
                "build_dependencies": ["pip"],
                "dependencies": [f"kedro_mlflow_tutorial=={PROJECT_VERSION}"],
            },
            signature="auto",
        ),
    )

    user_app_pipeline = create_user_app_pipeline()

    return {
        "etl_instances": etl_instances_pipeline,
        "etl_labels": etl_labels_pipeline,
        "training": training_pipeline_ml,
        "inference": inference_pipeline,
        "user_app": user_app_pipeline,
        "__default__": etl_instances_pipeline
        + etl_labels_pipeline
        + inference_pipeline
        + user_app_pipeline,
    }
