import pandas as pd
from sklearn.model_selection import train_test_split
import os

def read_txt(filepath):
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if ';' in line:
                text, emotion = line.strip().split(';', 1)
                rows.append({'text': text.strip(), 'emotion': emotion.strip().lower()})
    return pd.DataFrame(rows)

train1 = read_txt('data/raw/train.txt')
test1  = read_txt('data/raw/test.txt')
val1   = read_txt('data/raw/val.txt')

dataset1 = pd.concat([train1, test1, val1], ignore_index=True)
dataset1['source'] = 'dataset1'

try:
    dataset2 = pd.read_excel('data/raw/Emotion_final.xlsx')
except:
    dataset2 = pd.read_csv('data/raw/Emotion_final.csv')

text_cols = ['text', 'content', 'tweet', 'comment', 'utterance', 'sentence']
emotion_cols = ['emotion', 'sentiment', 'label', 'feeling']

text_col = next((c for c in dataset2.columns if c.lower() in text_cols), None)
emotion_col = next((c for c in dataset2.columns if c.lower() in emotion_cols), None)

if text_col is None or emotion_col is None:
    raise ValueError("Could not detect text/emotion columns in Dataset 2.")

dataset2 = dataset2[[text_col, emotion_col]]
dataset2.columns = ['text', 'emotion']
dataset2['emotion'] = dataset2['emotion'].str.lower()
dataset2['source'] = 'dataset2'

master_df = pd.concat([dataset1, dataset2], ignore_index=True)

emotion_map = {
    'happy': 'joy', 'happiness': 'joy', 'joyful': 'joy', 'love': 'joy',
    'sad': 'sadness', 'depressed': 'sadness',
    'angry': 'anger', 'rage': 'anger', 'annoyed': 'anger', 'disgust': 'anger',
    'scared': 'fear', 'fearful': 'fear',
    'surprised': 'surprise', 'shocked': 'surprise'
}

master_df['emotion'] = master_df['emotion'].replace(emotion_map)

master_df = master_df.drop_duplicates(subset='text').dropna().reset_index(drop=True)

print("Total samples:", len(master_df))
print("\nEmotion counts:\n", master_df['emotion'].value_counts())

train_df, temp_df = train_test_split(
    master_df, test_size=0.3, stratify=master_df['emotion'], random_state=42
)
test_df, val_df = train_test_split(
    temp_df, test_size=0.33, stratify=temp_df['emotion'], random_state=42
)

print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}, Val: {len(val_df)}")

os.makedirs("data/processed", exist_ok=True)

master_df.to_csv("data/processed/master_emotions_dataset.csv", index=False)

def save_txt(df, filename):
    with open(f"data/processed/{filename}", 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['text']};{row['emotion']}\n")

save_txt(train_df, "merged_train.txt")
save_txt(test_df,  "merged_test.txt")
save_txt(val_df,   "merged_val.txt")

print("\n All files saved successfully to data/processed/")
