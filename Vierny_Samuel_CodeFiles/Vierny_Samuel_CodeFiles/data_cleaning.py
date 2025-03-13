import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def data_cleaning():
    # Load Data
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    # Fill missing values
    df_train.loc[:, "keyword"] = df_train["keyword"].fillna("unknown")
    df_test.loc[:, "keyword"] = df_test["keyword"].fillna("unknown")
    df_train.loc[:, "location"] = df_train["location"].fillna("unknown")
    df_test.loc[:, "location"] = df_test["location"].fillna("unknown")

    # Text Preprocessing
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    #Clean and drop old text to avoid confusion
    df_train["clean_text"] = df_train["text"].apply(clean_text)
    df_test["clean_text"] = df_test["text"].apply(clean_text)
    df_train.drop(columns=["text"], inplace=True)
    df_test.drop(columns=["text"], inplace=True)

    #Add Length feature
    df_train["length"] = df_train["clean_text"].apply(lambda x : len(x))
    df_test["length"] = df_test["clean_text"].apply(lambda x : len(x))

    # Normalize 'length' Feature
    scaler = MinMaxScaler()
    df_train["length"] = scaler.fit_transform(df_train[["length"]])
    df_test["length"] = scaler.transform(df_test[["length"]])

    #Impute any missing values that might now arise after cleaning- and Ensure 'clean_text' is a string and replace NaN values
    df_train["clean_text"] = df_train["clean_text"].astype(str).fillna("")
    df_test["clean_text"] = df_test["clean_text"].astype(str).fillna("")

    print("Preprocessing Complete! Ready for Model Training.")
    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = data_cleaning()

    # Save processed datasets
    df_train.to_csv("data/cleaned_train.csv", index=False)
    df_test.to_csv("data/cleaned_test.csv", index=False)

    # Print column names and counts
    print("\n=== Train Dataset Columns & Count ===")
    print(df_train.columns.tolist())
    print(f"Total Columns: {len(df_train.columns)}")

    print("\n=== Test Dataset Columns & Count ===")
    print(df_test.columns.tolist())
    print(f"Total Columns: {len(df_test.columns)}")

    print("\nData Cleaning complete. Processed data saved as cleaned_test.csv.")