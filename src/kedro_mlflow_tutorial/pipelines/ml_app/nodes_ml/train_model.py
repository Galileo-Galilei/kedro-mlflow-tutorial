from sklearn.model_selection import train_test_split
import xgboost as xgb
import mlflow


def create_training_data(features, labels, test_size, random_state):
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def train_xgb_model(
    x_train,
    x_test,
    y_train,
    y_test,
    hyperparameters,
    num_boost_round,
    early_stopping_rounds,
):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    evaluation_list = [(dtrain, "train"), (dtest, "eval")]

    training_history = {}

    xgb_model = xgb.train(
        params=hyperparameters,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evaluation_list,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[xgb.callback.record_evaluation(training_history)],
    )

    # TODO: return metrics and log it with mlflow metrics dataset
    training_metrics = training_history["train"]
    training_metrics = {f"train_{k}": v for k, v in training_metrics.items()}

    for i, (metric_name, metric_history) in enumerate(training_metrics.items()):
        for metric in metric_history:
            mlflow.log_metric(metric_name, metric, step=i)

    eval_metrics = training_history["eval"]
    eval_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
    for i, (metric_name, metric_history) in enumerate(eval_metrics.items()):
        for metric in metric_history:
            mlflow.log_metric(metric_name, metric, step=i)

    return xgb_model


def plot_xgb_importance(model):
    # https://stackoverflow.com/questions/56151815/how-to-save-feature-importance-plot-of-xgboost-to-a-file-from-jupyter-notebook
    ax = xgb.plot_importance(model, max_num_features=20)
    ax.figure.tight_layout()
    return ax.figure


def calculate_contingency_matrix(xgb_model):
    pass

