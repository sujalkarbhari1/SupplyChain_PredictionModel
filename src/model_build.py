from flaml import AutoML
from config import Config

def train_flaml(X_train, y_train):
    automl = AutoML()

    settings = {
        "time_budget": 60,
        "metric": "accuracy",
        "task": "classification",
        "estimator_list": ["rf", "extra_tree", "xgboost", "lrl2"],
        "log_file_name": "flaml.log",
        "seed": 42
    }

    automl.fit(X_train=X_train, y_train=y_train, **settings)

    print("FLAML Training Completed")
    print("Best Model:", automl.model.estimator)

    return automl