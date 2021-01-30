from kedro.pipeline import Pipeline, node

from kedro_mlflow_tutorial.pipelines.etl_app.create_data import (
    create_instances,
    create_labels,
)


def create_etl_pipeline(**kwargs):
    pipeline_etl = Pipeline(
        [
            node(
                func=create_instances,
                inputs=dict(split="params:huggingface_split"),
                outputs="instances",
                tags="etl_instances",
            ),
            node(
                func=create_labels,
                inputs=dict(split="params:huggingface_split"),
                outputs="labels",
                tags="etl_labels",
            ),
        ]
    )

    return pipeline_etl
