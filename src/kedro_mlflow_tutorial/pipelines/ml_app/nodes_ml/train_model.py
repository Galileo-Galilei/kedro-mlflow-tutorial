import xgboost as xgb
from sklearn.model_selection import train_test_split


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

    hyperparameters = {**hyperparameters, "eval_metric": ["auc", "logloss"]}

    xgb_model = xgb.train(
        params=hyperparameters,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evaluation_list,
        evals_result=training_history,
        early_stopping_rounds=early_stopping_rounds,
    )

    return (
        xgb_model,
        training_history["train"]["auc"],
        training_history["eval"]["auc"],
        training_history["train"]["logloss"],
        training_history["eval"]["logloss"],
    )


def plot_xgb_importance(model):
    # https://stackoverflow.com/questions/56151815/how-to-save-feature-importance-plot-of-xgboost-to-a-file-from-jupyter-notebook
    ax = xgb.plot_importance(model, max_num_features=20)
    ax.figure.tight_layout()
    return ax.figure
