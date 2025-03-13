import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

class TweetTestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Flatten the tensor dimensions from (1, max_length) to just (max_length)
        item = {k: v.squeeze() for k, v in encoding.items()}
        return item

    def __len__(self):
        return len(self.texts)

def model_test_bert():
    # Load the BERT model and tokenizer
    model_path = "disaster_tweets_classification_best_bert_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the original test CSV with raw or cleaned text
    df_orig_test = pd.read_csv("data/processed_test.csv")

    # Replace any NaN with an empty string
    df_orig_test["clean_text"] = df_orig_test["clean_text"].fillna("")

    # Convert everything to string type
    df_orig_test["clean_text"] = df_orig_test["clean_text"].astype(str)

    test_texts = df_orig_test["clean_text"].tolist()  # or whatever column holds the text
    
    # Create dataset and dataloader
    test_dataset = TweetTestDataset(test_texts, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Inference
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        "id": df_orig_test["id"],
        "target": all_preds
    })
    
    # Save predictions
    submission.to_csv("submission_bert.csv", index=False)
    print("BERT submission saved to 'submission_bert.csv'.")

if __name__ == "__main__":
    model_test_bert()
