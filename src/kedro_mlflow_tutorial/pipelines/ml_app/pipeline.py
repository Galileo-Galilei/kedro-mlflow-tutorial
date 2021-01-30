from kedro.pipeline import Pipeline, node

from kedro_mlflow_tutorial.pipelines.ml_app.nodes_ml.predict import (
    decode_predictions,
    predict_with_xgboost,
)
from kedro_mlflow_tutorial.pipelines.ml_app.nodes_ml.preprocess_labels import (
    fit_label_encoder,
    transform_label_encoder,
)
from kedro_mlflow_tutorial.pipelines.ml_app.nodes_ml.preprocess_text import (
    convert_data_to_list,
    fit_tokenizer,
    lowerize_text,
    remove_punctuation,
    remove_stopwords,
    tokenize_text,
)
from kedro_mlflow_tutorial.pipelines.ml_app.nodes_ml.train_model import (
    create_training_data,
    plot_xgb_importance,
    train_xgb_model,
)


def create_ml_pipeline(**kwargs):
    pipeline_labels = Pipeline(
        [
            node(
                func=fit_label_encoder,
                inputs=dict(labels="labels"),
                outputs="label_encoder",
                tags=["training"],
            ),
            node(
                func=transform_label_encoder,
                inputs=dict(encoder="label_encoder", labels="labels"),
                outputs="encoded_labels",
                tags=["training"],
            ),
        ]
    )

    pipeline_text_processing = Pipeline(
        [
            node(
                func=lowerize_text,
                inputs=dict(data="instances"),
                outputs="text_lowered",
                tags=["training", "inference"],
            ),
            node(
                func=remove_stopwords,
                inputs=dict(data="text_lowered", stopwords="english_stopwords"),
                outputs="text_wo_stopwords",
                tags=["training", "inference"],
            ),
            node(
                func=remove_punctuation,
                inputs=dict(data="text_wo_stopwords"),
                outputs="cleaned_text",
                tags=["training", "inference"],
            ),
            node(
                func=convert_data_to_list,
                inputs=dict(data="cleaned_text"),
                outputs="formatted_text",
                tags=["training", "inference"],
            ),
            node(
                func=fit_tokenizer,
                inputs=dict(list_data="formatted_text", num_words="params:vocab_size"),
                outputs="keras_tokenizer",
                tags=["training"],
            ),
            node(
                func=tokenize_text,
                inputs=dict(tokenizer="keras_tokenizer", list_data="formatted_text"),
                outputs="one_hot_encoded_data",
                tags=["training", "inference"],
            ),
        ]
    )

    pipeline_training = Pipeline(
        [
            node(
                func=create_training_data,
                inputs=dict(
                    features="one_hot_encoded_data",
                    labels="encoded_labels",
                    test_size="params:early_stopping_test_size",
                    random_state="params:split_seed",
                ),
                outputs=["x_train", "x_test", "y_train", "y_test"],
                tags=["training"],
            ),
            node(
                func=train_xgb_model,
                inputs=dict(
                    x_train="x_train",
                    x_test="x_test",
                    y_train="y_train",
                    y_test="y_test",
                    hyperparameters="params:xgb_hyperparameters",
                    num_boost_round="params:xgb_num_rounds",
                    early_stopping_rounds="params:xgb_early_stopping_rounds",
                ),
                outputs="xgb_model",
                tags=["training"],
            ),
            node(
                func=plot_xgb_importance,
                inputs=dict(model="xgb_model"),
                outputs="xgb_feature_importance",
                tags=["training"],
            ),
        ]
    )

    pipeline_inference = Pipeline(
        [
            node(
                func=predict_with_xgboost,
                inputs=dict(model="xgb_model", data="one_hot_encoded_data"),
                outputs="xgb_predictions",
                tags=["inference"],
            ),
            node(
                func=decode_predictions,
                inputs=dict(encoder="label_encoder", data="xgb_predictions"),
                outputs="final_predictions",
                tags=["inference"],
            ),
        ]
    )

    return (
        pipeline_labels
        + pipeline_text_processing
        + pipeline_training
        + pipeline_inference
    )
