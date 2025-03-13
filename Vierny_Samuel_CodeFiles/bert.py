# bert.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from tfidf_vectorizer import split_and_apply_tfidf
from config import PYCARET_CONFIG, DEEP_LEARNING_CONFIG

# A simple Dataset for our tweets
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_bert_model(cfg, deep_cfg):
    # Check device and print it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processed training and test data
    df = pd.read_csv("data/processed_train.csv")
    df_test = pd.read_csv("data/processed_test.csv")
    
    # Use the TF-IDF function to get the train/validation split indices.
    # (We only use the indices here so we can retrieve the original text for BERT.)
    X_train_tfidf, X_val_tfidf, y_train, y_val = split_and_apply_tfidf(
        df,
        test_df=df_test,
        target=cfg["target"],
        text_column="clean_text",
        max_features=1000,
        ngram_range=(1, 1),
        test_size=0.2,
        random_state=cfg["session_id"]
    )
    
    # Retrieve the corresponding texts for training and validation using the indices.
    train_indices = X_train_tfidf.index
    val_indices = X_val_tfidf.index
    train_texts = df.loc[train_indices, "clean_text"].tolist()
    val_texts = df.loc[val_indices, "clean_text"].tolist()
    
    # Convert labels to list
    train_labels = y_train.tolist()
    val_labels = y_val.tolist()
    
    # Initialize BERT tokenizer and model (for sequence classification).
    # Here we assume a binary or multi-class classification based on the unique labels.
    num_labels = len(set(train_labels))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    
    # Move the model to the selected device (GPU if available)
    model.to(device)
    
    # Tokenize the texts with truncation and padding.
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    # Create Dataset objects
    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)
    
    # Retrieve BERT-specific training parameters from config
    epochs = deep_cfg["deep_model_config"]["bert"]["epochs"]
    batch_size = deep_cfg["deep_model_config"]["bert"]["batch_size"]
    
    # Set up training arguments, enabling fp16 if GPU is available
    training_args = TrainingArguments(
        output_dir="./bert_output",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        logging_dir="./bert_logs",
        logging_steps=10,
        seed=cfg["session_id"],
        fp16=torch.cuda.is_available()  # Enables mixed precision training if GPU is available
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the BERT model
    trainer.train()
    
    # Save the final (best) BERT model and tokenizer.
    model_save_path = cfg["experiment_name"] + "_best_bert_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Best BERT model saved to '{model_save_path}'.")

if __name__ == "__main__":
    train_bert_model(PYCARET_CONFIG, DEEP_LEARNING_CONFIG)
