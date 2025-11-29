import pandas as pd
import re
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from tqdm import tqdm

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"[^a-zA-Z0-9\s:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  

id2label = model.config.id2label

def predict_emotions_batch(text_list, batch_size=16, device=None):
    """Predict emotions in batches for faster processing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predictions = []

    for i in tqdm(range(0, len(text_list), batch_size), desc="Processing batches"):
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            labels = torch.argmax(probs, dim=1).tolist()
            all_predictions.extend([id2label[l] for l in labels])

    return all_predictions

def load_input_file(path="data/processed/master_emotions_dataset.csv"):
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    text_cols = ["text", "comment", "content", "body", "utterance", "sentence"]
    timestamp_cols = ["timestamp", "time", "date"]

    text_col = next((c for c in df.columns if c.lower() in text_cols), None)
    time_col = next((c for c in df.columns if c.lower() in timestamp_cols), None)

    if text_col is None:
        raise ValueError(" No valid text/comment column found.")

    return df, text_col, time_col

def run_pipeline():
    print("\n Loading input dataset...")
    df, text_col, time_col = load_input_file()

    print(f" Detected text column: {text_col}")
    if time_col:
        print(f" Detected timestamp column: {time_col}")
    else:
        print(" No timestamp found, continuing without timestamps.")

    print("\n Cleaning text...")
    df["cleaned_text"] = df[text_col].apply(clean_text)

    print("\n Running emotion classification...")
    df["predicted_emotion"] = predict_emotions_batch(df["cleaned_text"].tolist(), batch_size=16)

    output_cols = []
    if time_col:
        output_cols.append(time_col)
    output_cols += [text_col, "cleaned_text", "predicted_emotion"]

    final_df = df[output_cols]

    os.makedirs("data/outputs", exist_ok=True)
    final_df.to_csv("data/outputs/final_emotions_output.csv", index=False)

    print("\n DONE! Final file saved at:")
    print(" data/outputs/final_emotions_output.csv")

if __name__ == "__main__":
    run_pipeline()
