import pandas as pd
from pycaret.classification import load_model

def model_test():
    # Load the saved best PyCaret model.
    model = load_model("disaster_tweets_classification_best_pycaret_model")
    print("Model loaded successfully.")
    
    # Load the TF-IDF transformed test data (features only).
    df_tfidf = pd.read_csv("data/processed_test_tfidf.csv")
    print("TF-IDF test features loaded.")
    
    # Directly use the underlying model's predict method.
    # This bypasses PyCaret's internal feature alignment.
    test_predictions = model.predict(df_tfidf)
    print("Predictions generated.")
    
    # Load the original processed test data to retrieve the 'id' column.
    df_orig_test = pd.read_csv("data/processed_test.csv")
    
    # Create the submission DataFrame using the original 'id' and the predicted target.
    submission = pd.DataFrame({
         "id": df_orig_test["id"],
         "target": test_predictions
    })
    
    # Save the submission file.
    submission.to_csv("submission.csv", index=False)
    print("Submission file saved to 'submission.csv'.")

if __name__ == "__main__":
    model_test()
