import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
df = pd.read_csv("data/outputs/final_emotions_output.csv")

st.title("Emotion Analysis Dashboard")


st.header("Overview")
st.write(f"Total comments: {len(df)}")
st.write(f"Most common emotion: {df['predicted_emotion'].mode()[0]}")

fig_pie = px.pie(df, names='predicted_emotion', title="Emotion Distribution")
st.plotly_chart(fig_pie)

st.header("Emotion Heatmap")
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
else:
    df['hour'] = 0

heatmap_data = df.groupby(['predicted_emotion','hour']).size().unstack(fill_value=0)
fig_heatmap = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd")
st.plotly_chart(fig_heatmap)

st.header("Top Comments per Emotion")
selected_emotion = st.selectbox("Select Emotion", df['predicted_emotion'].unique())
top_comments = df[df['predicted_emotion']==selected_emotion]['cleaned_text'].head(10)
for i, comment in enumerate(top_comments, 1):
    st.write(f"{i}. {comment}")

st.header("Word Clouds per Emotion")
for emotion in df['predicted_emotion'].unique():
    text = " ".join(df[df['predicted_emotion']==emotion]['cleaned_text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(emotion, fontsize=16)
    st.pyplot(fig)
