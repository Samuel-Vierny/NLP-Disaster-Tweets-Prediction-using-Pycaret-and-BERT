# config.py

PYCARET_CONFIG = {
    "Train_trad_ml": True,
    # General experiment settings
    "experiment_name": "disaster_tweets_classification",
    "target": "target",  # Target column name in your dataset
    "fold": 5,
    "session_id": 42,
    "verbose": True,
    "n_jobs": -1,  # Use all cores

    # Models to train in PyCaret (use PyCaret's model abbreviations)
    "models_to_train": ["lr", "dt", "rf", "xgboost", "svm", "nb", "knn", "et"],

    # Tuning settings
    "tune_models": True,
    "optimize_metric": "F1",
}

DEEP_LEARNING_CONFIG = {
    # Settings for deep learning models (placeholders for future use)
    "train_deep_models": False,
    "deep_model_config": {
        "bert": {
            "epochs": 5,
            "batch_size": 128,
        },
    },
}