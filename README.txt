# Disaster Tweets Classification

This repository contains scripts to preprocess, feature engineer, and build predictive models for disaster tweets using both traditional machine learning (via PyCaret) and deep learning (using BERT).

## File Overview

- **download_data.py**  
  Downloads and extracts the dataset from Kaggle into the `data/` directory.

- **data_cleaning.py**  
  Cleans raw data by filling missing values, normalizing text (removing URLs, mentions, punctuation, etc.), and generating a new `clean_text` column along with basic features (e.g., text length).

- **data_manipulation.py**  
  Enhances the cleaned data by adding text-based features (hashtag, mention, URL counts) and performing sentiment analysis using TextBlob. Saves the processed train and test sets.

- **tfidf_vectorizer.py**  
  Contains the `split_and_apply_tfidf` function that converts text into TF-IDF features, applies variance thresholding, and saves the transformed test features.

- **config.py**  
  Holds configuration settings for both PyCaret experiments and deep learning (BERT) training, such as experiment names, target column, session IDs, epochs, and batch sizes.

- **pycaret_ml.py**  
  Sets up and runs a PyCaret experiment to train, tune, and compare multiple traditional ML models using the TF-IDF features.

- **bert.py**  
  Trains a BERT-based sequence classifier using preprocessed tweet data. It handles tokenization, model training with Hugging Face’s Trainer, and saves the best model.

- **model_test_bert.py**  
  Loads the trained BERT model to generate predictions on the test set and saves a submission file.

- **model_test_pycaret.py**  
  Loads the best PyCaret model, performs predictions on TF-IDF transformed test data, and saves the submission CSV.

## How to Run

1. **Data Preparation:**
   - Run `download_data.py` to download and extract the dataset.
   - Execute `data_cleaning.py` to clean the raw data and save cleaned CSV files.
   - Execute `data_manipulation.py` to add additional features and save the processed datasets.

2. **Feature Engineering:**
   - The `tfidf_vectorizer.py` script is used within both the PyCaret and BERT pipelines to transform text data into TF-IDF features.

3. **Model Training:**
   - **Traditional ML (PyCaret):**  
     Run `pycaret_ml.py` to set up the experiment, train multiple models, and save the best performing model.
   - **Deep Learning (BERT):**  
     Run `bert.py` to train a BERT classifier on the processed tweets and save the final model.

4. **Model Testing & Submission:**
   - Run `model_test_bert.py` to generate predictions using the trained BERT model.
   - Run `model_test_pycaret.py` to generate predictions using the best PyCaret model.

## Requirements

- **Python 3.x**  
- **Libraries:** pandas, torch, transformers, scikit-learn, nltk, kaggle, pycaret, textblob, etc.  
- A GPU is recommended for training BERT (though CPU mode is supported).

## Notes

- Ensure the `data/` directory exists and all data files are in place before running the scripts.
- Adjust configuration settings in `config.py` as needed for your experiment.
- Verify that the Kaggle API is configured properly for data download.

