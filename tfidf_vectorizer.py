import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

def split_and_apply_tfidf(train_df, test_df, target, text_column="clean_text", 
                          max_features=5000, ngram_range=(1, 2), test_size=0.2, random_state=42):
    """
    Transforms text data into TF-IDF features and applies variance thresholding.
    
    - Splits train_df into training and validation sets.
    - Fits the TF-IDF vectorizer and variance selector on the training set.
    - Transforms the validation and test sets using the same fitted pipeline.
    - Exports the processed test set as 'data/processed_test_tfidf.csv'.
    
    Parameters:
      - train_df: DataFrame containing training data (with the target column)
      - test_df: DataFrame containing test data (without the target column)
      - target: Name of the target column in train_df.
      - text_column: Column name containing text to transform.
      - max_features: Maximum number of features for the TF-IDF vectorizer.
      - ngram_range: N-gram range for the TF-IDF vectorizer.
      - test_size: Fraction of train_df to use as the validation set.
      - random_state: Random state for the train/validation split.
      
    Returns:
      - X_train_selected_df: DataFrame of transformed training features.
      - X_val_selected_df: DataFrame of transformed validation features.
      - y_train: Series of target values for the training set.
      - y_val: Series of target values for the validation set.
    """
    # Fill missing text values
    train_df[text_column] = train_df[text_column].fillna("")
    test_df[text_column] = test_df[text_column].fillna("")
    
    # Split the training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_df, train_df[target], test_size=test_size, random_state=random_state)
    
    # Initialize and fit TF-IDF on the training text only
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train[text_column])
    
    # Transform the validation and test data using the fitted vectorizer
    X_val_tfidf = vectorizer.transform(X_val[text_column])
    X_test_tfidf = vectorizer.transform(test_df[text_column])
    
    # Convert sparse matrices to DataFrames using the same feature names
    X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), 
                                    columns=vectorizer.get_feature_names_out(), 
                                    index=X_train.index)
    X_val_tfidf_df = pd.DataFrame(X_val_tfidf.toarray(), 
                                  columns=vectorizer.get_feature_names_out(), 
                                  index=X_val.index)
    X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), 
                                   columns=vectorizer.get_feature_names_out(), 
                                   index=test_df.index)
    
    # Apply VarianceThreshold to remove low variance features (fit on training data)
    selector = VarianceThreshold(threshold=0.0005)
    X_train_selected = selector.fit_transform(X_train_tfidf_df)
    selected_features = X_train_tfidf_df.columns[selector.get_support()]
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
    
    # Transform validation and test data using the same selector
    X_val_selected = selector.transform(X_val_tfidf_df)
    X_val_selected_df = pd.DataFrame(X_val_selected, columns=selected_features, index=X_val.index)
    X_test_selected = selector.transform(X_test_tfidf_df)
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features, index=test_df.index)
    
    # Export the processed test data to CSV
    X_test_selected_df.to_csv("data/processed_test_tfidf.csv", index=False)
    print("Test TF-IDF features saved to 'data/processed_test_tfidf.csv'.")
    
    return X_train_selected_df, X_val_selected_df, y_train, y_val

if __name__ == "__main__":
    # Load your cleaned training and test data
    df_train = pd.read_csv("data/processed_train.csv")
    df_test = pd.read_csv("data/processed_test.csv")
    
    # Call the function with both DataFrames
    X_train, X_val, y_train, y_val = split_and_apply_tfidf(df_train, df_test, target="target", text_column="clean_text")
    
    print("Train X shape:", X_train.shape)
    print("Validation X shape:", X_val.shape)
    print("Train y shape:", y_train.shape)
    print("Validation y shape:", y_val.shape)
