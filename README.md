# AI Fake Review Detection System

A production-ready Streamlit web application for detecting fake or suspicious product reviews using Natural Language Processing and Machine Learning.

The app combines a trained Logistic Regression model, TF-IDF vectorization, NLTK preprocessing, sentiment analysis, and rule-based spam detection to deliver a polished portfolio-ready AI experience.

## Features

- Modern Streamlit SaaS-style interface
- Product review text analysis
- NLP preprocessing with lowercase conversion, special character cleanup, stopword removal, and whitespace normalization
- TF-IDF vectorization
- Logistic Regression inference using saved model artifacts
- Rule-based suspicious review detection
- Confidence score and progress meter
- Sentiment analysis using TextBlob
- Word count, review quality score, and spam indicators
- Professional sidebar with project, model, dataset, and developer information
- Streamlit Cloud and GitHub deployment-ready structure

## Project Structure

```text
Fake-Review-Detection/
├── app.py
├── requirements.txt
├── README.md
├── fake_review_model.pkl
├── tfidf_vectorizer.pkl
├── final_fake_review_dataset.csv
├── assets/
│   └── styles.css
```

Note: this app also supports `realistic_fake_review_dataset_5000.csv` as a fallback if `final_fake_review_dataset.csv` is not present.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLTK
- TextBlob
- Joblib
- Pandas

## Screenshots

Add screenshots after running the app locally:

```text
screenshots/
├── home-page.png
├── genuine-result.png
└── fake-result.png
```

## Installation Guide

1. Clone the repository or open the project folder.

```bash
git clone <your-repository-url>
cd Fake-Review-Detection
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app.

```bash
streamlit run app.py
```

## Required Model Files

Keep these files in the project root:

- `fake_review_model.pkl`
- `tfidf_vectorizer.pkl`
- `final_fake_review_dataset.csv`

The model should be a trained Logistic Regression classifier, and the vectorizer should be the TF-IDF vectorizer used during training.

## How It Works

1. The user enters a product review.
2. Rule-based spam checks scan for repeated words, promotional language, gibberish text, and very short reviews.
3. If the review is suspicious, the app shows a warning without ML prediction.
4. If the review passes validation, text is preprocessed using NLTK.
5. The TF-IDF vectorizer transforms the review into numerical features.
6. The Logistic Regression model predicts whether the review is genuine or fake.
7. The UI displays the prediction, confidence score, sentiment, quality score, and NLP insights.

## Deployment Guide

### Streamlit Cloud

1. Push the project to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app and connect your GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

### GitHub Checklist

- Include `app.py`
- Include `requirements.txt`
- Include `README.md`
- Include `assets/styles.css`
- Include model and vectorizer pickle files
- Include dataset file if allowed by size and license

## Future Improvements

- Add model retraining pipeline
- Add explainable AI with top TF-IDF terms
- Add batch CSV review uploads
- Add user authentication for dashboard usage
- Add review history and analytics
- Add transformer-based NLP model comparison

## Local Run Command

```bash
streamlit run app.py
```
