📌 Sentimentmap

A lightweight NLP project that analyzes social media comments and maps them to emotions such as joy, anger, fear, sadness, disgust, and surprise.
The results are visualized through heatmaps, charts, and wordclouds, and served through a small Streamlit dashboard.

✨ What This Project Does

Cleans and preprocesses raw text

Classifies comments using a transformer-based emotion model

Generates visual summaries (heatmap, distributions, wordclouds)

Saves all outputs in structured folders

Provides a simple Streamlit UI for exploring the results

Useful for social-media analytics, sentiment research, and understanding user reactions at scale.

▶️ How to Run Locally
1. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

2. Install dependencies
pip install -r requirements.txt

3. Run preprocessing + emotion classification
python src/run_preprocessing_and_emotion_classification.py

4. Launch Streamlit app
streamlit run src/app.py

🌐 Live Demo

https://sentimentmap-cdsee5lv2leuxwrvutgiu9.streamlit.app/
