# ----- pycaret_ml.py -----
import pandas as pd
from pycaret.classification import setup, create_model, tune_model, compare_models, save_model, pull, predict_model
from config import PYCARET_CONFIG
from tfidf_vectorizer import split_and_apply_tfidf  # Ensure this function is defined as in the previous example

def run_pycaret_experiment(cfg):
    # Load cleaned training data
    df = pd.read_csv("data/processed_train.csv")
    
    # Split the data and apply TF-IDF transformation
    train_x, val_x, train_y, val_y = split_and_apply_tfidf(
        df, 
        target=cfg["target"], 
        text_column="clean_text", 
        max_features=1000, 
        ngram_range=(1, 1), 
        test_size=0.2, 
        random_state=cfg["session_id"],
        test_df=pd.read_csv("data/processed_test.csv")
    )
    
    # Combine TF-IDF features with target for PyCaret
    train_df = train_x.copy()
    train_df[cfg["target"]] = train_y.values  # Ensure proper alignment
    
    val_df = val_x.copy()
    val_df[cfg["target"]] = val_y.values

    # Initialize the PyCaret experiment on the training DataFrame
    exp = setup(
        data=train_df,
        target=cfg["target"],
        fold=cfg["fold"],
        session_id=cfg["session_id"],
        verbose=cfg["verbose"],
        n_jobs=cfg["n_jobs"],
        categorical_imputation="Unknown"
    )
    
    print("\nExperiment setup complete.")
    print(pull())
    
    # Dictionary to store trained models
    trained_models = {}
    
    # Create and (optionally) tune models using training data only
    for model_code in cfg["models_to_train"]:
        print(f"\nCreating model: {model_code}")
        model = create_model(model_code)
        if cfg["tune_models"]:
            print(f"Tuning model: {model_code}")
            model = tune_model(model, optimize=cfg["optimize_metric"])
        trained_models[model_code] = model
        print(f"Model {model_code} complete.")
        print(pull())
    
    # Compare models based on cross-validation performance on training data
    best_model = compare_models(include=list(trained_models.values()), sort=cfg["optimize_metric"])
    print("\nBest Model from training CV:")
    print(best_model)
    
    # Evaluate best model on the external validation set
    val_results = predict_model(best_model, data=val_df)
    print("\nExternal Validation Performance (sample predictions):")
    print(val_results.head())
    
    # Optional: Fine tune the best model further (using PyCaret's tune_model) based on CV metrics
    tuned_best_model = tune_model(best_model, optimize=cfg["optimize_metric"])
    print("\nTuned Best Model:")
    print(tuned_best_model)
    
    # Save the final best model
    model_filename = cfg["experiment_name"] + "_best_pycaret_model"
    save_model(tuned_best_model, model_filename)
    print(f"\nBest model saved as '{model_filename}.pkl'")
    
    return trained_models, tuned_best_model

if __name__ == "__main__":
    trained_models, best_model = run_pycaret_experiment(PYCARET_CONFIG)
