import re
from collections import Counter
from pathlib import Path

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from textblob import TextBlob


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_review_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
CSV_DATASET_PATH = BASE_DIR / "final_fake_review_dataset.csv"
FALLBACK_DATASET_PATH = BASE_DIR / "realistic_fake_review_dataset_5000.csv"
STYLE_PATH = BASE_DIR / "assets" / "styles.css"

SPAM_KEYWORDS = {
    "amazing",
    "perfect",
    "best",
    "excellent",
    "highly recommended",
}


st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Load custom CSS if the assets file exists."""
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_nltk_stopwords() -> set[str]:
    """Load NLTK stopwords, downloading the corpus on first run if needed."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model and vectorizer from disk."""
    if not MODEL_PATH.exists():
        st.error("Model file not found: fake_review_model.pkl")
        st.stop()
    if not VECTORIZER_PATH.exists():
        st.error("Vectorizer file not found: tfidf_vectorizer.pkl")
        st.stop()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data(show_spinner=False)
def load_dataset_preview() -> tuple[int | None, list[str]]:
    """Load lightweight dataset information for the sidebar."""
    path = CSV_DATASET_PATH if CSV_DATASET_PATH.exists() else FALLBACK_DATASET_PATH
    if not path.exists():
        return None, []

    try:
        if path.suffix.lower() == ".csv":
            data = pd.read_csv(path, nrows=5)
            row_count = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
        else:
            data = pd.read_excel(path, nrows=5)
            row_count = None
        return row_count, list(data.columns)
    except Exception:
        return None, []


def preprocess_text(text: str) -> str:
    """Normalize review text before vectorization."""
    stop_words = load_nltk_stopwords()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)


def detect_spam_signals(text: str) -> list[str]:
    """Return rule-based spam indicators found in a review."""
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = cleaned.split()
    indicators: list[str] = []

    if len(words) < 4:
        indicators.append("Review is extremely short")

    word_counts = Counter(words)
    repeated = [word for word, count in word_counts.items() if count >= 3 and len(word) > 2]
    if repeated:
        indicators.append(f"Repeated words detected: {', '.join(repeated[:3])}")

    keyword_hits = [keyword for keyword in SPAM_KEYWORDS if keyword in cleaned]
    if len(keyword_hits) >= 2:
        indicators.append(f"Promotional language detected: {', '.join(keyword_hits[:4])}")

    alpha_ratio = sum(char.isalpha() for char in text) / max(len(text), 1)
    unique_ratio = len(set(words)) / max(len(words), 1)
    long_random_tokens = [word for word in words if len(word) > 14 and not re.search(r"[aeiou]", word)]

    if alpha_ratio < 0.55 or long_random_tokens or (len(words) >= 6 and unique_ratio < 0.35):
        indicators.append("Possible gibberish or low-quality text pattern")

    return indicators


def sentiment_label(polarity: float) -> str:
    if polarity > 0.2:
        return "Positive"
    if polarity < -0.2:
        return "Negative"
    return "Neutral"


def review_quality_score(text: str, spam_indicators: list[str]) -> int:
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    word_count = len(words)
    unique_ratio = len(set(word.lower() for word in words)) / max(word_count, 1)
    sentence_count = max(len(re.findall(r"[.!?]+", text)), 1)
    avg_sentence_length = word_count / sentence_count

    score = 50
    score += min(word_count, 80) * 0.35
    score += unique_ratio * 25
    if 8 <= avg_sentence_length <= 28:
        score += 12
    score -= len(spam_indicators) * 18

    return int(max(0, min(100, score)))


def normalize_prediction(raw_prediction) -> str:
    """Map common model labels to user-facing labels."""
    value = str(raw_prediction).strip().lower()
    fake_values = {"fake", "suspicious", "spam", "1", "false", "deceptive"}
    genuine_values = {"genuine", "real", "truthful", "0", "true", "authentic"}
    if value in fake_values:
        return "Fake/Suspicious Review"
    if value in genuine_values:
        return "Genuine Review"
    return "Fake/Suspicious Review" if value in {"1.0"} else "Genuine Review"


def get_prediction_confidence(model, vectorized_review, prediction) -> float:
    """Return confidence for the predicted class."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_review)[0]
        if hasattr(model, "classes_"):
            class_list = list(model.classes_)
            try:
                index = class_list.index(prediction)
                return float(probabilities[index])
            except ValueError:
                return float(max(probabilities))
        return float(max(probabilities))

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(vectorized_review)[0])
        return 1 / (1 + pow(2.71828, -abs(score)))

    return 0.75


def result_card(kind: str, title: str, confidence: float, explanation: str) -> None:
    st.markdown(
        f"""
        <div class="result-card {kind}">
            <div class="result-label">{title}</div>
            <div class="result-score">{confidence:.1%}</div>
            <p>{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    row_count, columns = load_dataset_preview()

    with st.sidebar:
        st.markdown("## Review Lab")
        st.markdown(
            "A portfolio-ready NLP application that combines machine learning "
            "with rule-based review quality checks."
        )

        st.markdown("### Technologies Used")
        st.markdown("Python · Streamlit · Scikit-learn · NLTK · TextBlob · Joblib")

        st.markdown("### Model Information")
        st.markdown(
            "Classifier: Logistic Regression  \n"
            "Features: TF-IDF vectors  \n"
            "Preprocessing: lowercase, cleanup, stopword removal"
        )

        st.markdown("### Dataset Info")
        if row_count:
            st.markdown(f"Records detected: **{row_count:,}**")
        else:
            st.markdown("Dataset file detected and loaded where available.")
        if columns:
            st.markdown(f"Columns: `{', '.join(columns[:6])}`")

        st.markdown("### Developer Info")
        st.markdown("Name: Nikita Adhau"n/)


def main() -> None:
    load_css()
    model, vectorizer = load_artifacts()
    render_sidebar()

    st.markdown(
        """
        <section class="hero">
            <div class="hero-badge">NLP + Machine Learning</div>
            <h1>Fake Review Detection System</h1>
            <p>
                Analyze product reviews with a hybrid engine that blends
                TF-IDF, Logistic Regression, sentiment signals, and spam rules.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    review_text = st.text_area(
        "Enter product review",
        placeholder="Example: The product quality is good, delivery was on time, and the packaging felt reliable.",
        height=170,
        label_visibility="collapsed",
    )

    analyze = st.button("Analyze Review", use_container_width=True, type="primary")

    if analyze:
        if not review_text or not review_text.strip():
            result_card(
                "invalid",
                "Invalid Input",
                0,
                "Please enter a product review before running the analysis.",
            )
            return

        with st.spinner("Analyzing review authenticity..."):
            spam_indicators = detect_spam_signals(review_text)
            word_count = len(re.findall(r"\b[a-zA-Z]+\b", review_text))
            polarity = TextBlob(review_text).sentiment.polarity
            quality = review_quality_score(review_text, spam_indicators)

            if spam_indicators:
                result_card(
                    "invalid",
                    "Suspicious Review Pattern",
                    max(0.15, 1 - quality / 100),
                    "Rule-based checks found suspicious signals, so ML prediction was skipped.",
                )
            else:
                processed_review = preprocess_text(review_text)
                vectorized_review = vectorizer.transform([processed_review])
                prediction = model.predict(vectorized_review)[0]
                label = normalize_prediction(prediction)
                confidence = get_prediction_confidence(model, vectorized_review, prediction)

                if label == "Genuine Review":
                    result_card(
                        "genuine",
                        "Genuine Review",
                        confidence,
                        "The review appears authentic based on the model and NLP quality checks.",
                    )
                else:
                    result_card(
                        "fake",
                        "Fake/Suspicious Review",
                        confidence,
                        "The review shows patterns commonly associated with suspicious or fake reviews.",
                    )

        st.markdown("### NLP Insights")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Word Count", word_count)
        metric_cols[1].metric("Sentiment", sentiment_label(polarity), f"{polarity:.2f}")
        metric_cols[2].metric("Quality Score", f"{quality}/100")
        metric_cols[3].metric("Spam Signals", len(spam_indicators))

        st.markdown("#### Confidence / Authenticity Meter")
        st.progress(quality / 100)

        if spam_indicators:
            st.markdown("#### Spam Indicators")
            for indicator in spam_indicators:
                st.warning(indicator)
        else:
            st.success("No major rule-based spam indicators were detected.")

    st.markdown(
        """
        <footer>
            <span>AI Fake Review Detection System</span>
            <span>Built with Streamlit, NLP, and Machine Learning</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
