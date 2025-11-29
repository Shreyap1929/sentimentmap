import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

csv_path = "data/outputs/final_emotions_output.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found at {csv_path}")

df = pd.read_csv(csv_path)

if 'predicted_emotion' not in df.columns:
    raise ValueError("Column 'predicted_emotion' not found in CSV")

print("Loaded CSV successfully!")
print("Emotion counts:")
print(df['predicted_emotion'].value_counts())

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour  
else:
    df['hour'] = 0  

heatmap_data = df.groupby(['predicted_emotion', 'hour']).size().unstack(fill_value=0)

plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd")
plt.title("Emotion Heatmap by Hour")
plt.ylabel("Emotion")
plt.xlabel("Hour")
plt.tight_layout()
os.makedirs("data/outputs", exist_ok=True)
plt.savefig("data/outputs/emotion_heatmap.png")
plt.show()
print(" Emotion Heatmap saved: data/outputs/emotion_heatmap.png")

trend_data = df.groupby(['hour','predicted_emotion']).size().unstack(fill_value=0)

trend_data.plot(figsize=(12,6), marker='o')
plt.title("Emotion Trends Over Time")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/outputs/emotion_trends.png")
plt.show()
print(" Trend Line Chart saved: data/outputs/emotion_trends.png")

df['predicted_emotion'].value_counts().plot.pie(
    autopct='%1.1f%%', figsize=(6,6), cmap='Set3', legend=False)
plt.ylabel("")
plt.title("Emotion Distribution")
plt.savefig("data/outputs/emotion_pie.png")
plt.show()
print("Emotion Pie Chart saved: data/outputs/emotion_pie.png")

emotions = df['predicted_emotion'].unique()

for emotion in emotions:
    text = " ".join(df[df['predicted_emotion']==emotion]['cleaned_text'].astype(str).tolist())
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='tab10'
    ).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud — {emotion}")
    plt.tight_layout()
    plt.savefig(f"data/outputs/wordcloud_{emotion}.png")
    plt.show()
    print(f"Word Cloud saved: data/outputs/wordcloud_{emotion}.png")

top_comments = df.groupby('predicted_emotion')['cleaned_text'].apply(lambda x: x.head(5))
top_comments_csv = "data/outputs/top_comments_per_emotion.csv"
top_comments.to_csv(top_comments_csv)
print(f"Top comments per emotion saved: {top_comments_csv}")
print(top_comments)
