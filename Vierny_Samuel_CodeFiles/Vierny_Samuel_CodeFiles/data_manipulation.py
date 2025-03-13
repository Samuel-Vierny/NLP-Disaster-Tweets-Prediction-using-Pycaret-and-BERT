# ----- data_manipulation.py -----
import pandas as pd
from textblob import TextBlob

def data_manipulation():
    # Call data_cleaning output from cleaned CSV files
    df_train = pd.read_csv("data/cleaned_train.csv")
    df_test = pd.read_csv("data/cleaned_test.csv")
    
    # Ensure 'clean_text' is a string and replace NaN values just in case
    df_train["clean_text"] = df_train["clean_text"].astype(str).fillna("")
    df_test["clean_text"] = df_test["clean_text"].astype(str).fillna("")

    # Add text-based features (hashtags, mentions, URLs) to both datasets
    df_train["hashtag_count"] = df_train["clean_text"].apply(lambda x: x.count("#"))
    df_test["hashtag_count"] = df_test["clean_text"].apply(lambda x: x.count("#"))

    df_train["mention_count"] = df_train["clean_text"].apply(lambda x: x.count("@"))
    df_test["mention_count"] = df_test["clean_text"].apply(lambda x: x.count("@"))

    df_train["url_count"] = df_train["clean_text"].apply(lambda x: x.count("http"))
    df_test["url_count"] = df_test["clean_text"].apply(lambda x: x.count("http"))

    # Sentiment Analysis on Disaster Tweets
    df_train['polarity'] = df_train['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_train['subjectivity'] = df_train['clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    df_test['polarity'] = df_test['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_test['subjectivity'] = df_test['clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = data_manipulation()
    
    # Save processed datasets
    df_train.to_csv("data/processed_train.csv", index=False)
    df_test.to_csv("data/processed_test.csv", index=False)

    # Print column names and counts
    print("\n=== Train Dataset Columns & Count ===")
    print(df_train.columns.tolist())
    print(f"Total Columns: {len(df_train.columns)}")

    print("\n=== Test Dataset Columns & Count ===")
    print(df_test.columns.tolist())
    print(f"Total Columns: {len(df_test.columns)}")

    print("\nData manipulation complete. Processed data saved as processed_train.csv.")
