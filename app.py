import streamlit as st
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS

from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import pandas as pd

# ------------------------------------------------------
# Download NLTK Dependencies
# ------------------------------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ------------------------------------------------------
# Preprocessing for Extractive Step
# ------------------------------------------------------
def preprocess_text(text):
    sentences = sent_tokenize(text)
    clean = []

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    for s in sentences:
        s2 = re.sub(r"[^a-zA-Z0-9\s]", "", s.lower())
        tokens = word_tokenize(s2)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        clean.append(" ".join(tokens))

    return sentences, clean

# ------------------------------------------------------
# Extractive Summarizer (LexRank)
# ------------------------------------------------------
def extractive_summary(sentences):
    lxr = LexRank(sentences, stopwords=STOPWORDS)
    summary = lxr.get_summary(sentences, summary_size=4)
    return " ".join(summary)

# ------------------------------------------------------
# Load Abstractive Model (Cached)
# ------------------------------------------------------
@st.cache_resource
def load_bart():
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    return model, tokenizer

# ------------------------------------------------------
# Abstractive Summarization
# ------------------------------------------------------
def abstractive_summary(text, tone):
    model, tokenizer = load_bart()

    prompt = (
        f"Summarize this text in a detailed and natural way. "
        f"Tone: {tone}. Text: {text}"
    )

    inputs = tokenizer([prompt], max_length=1024, truncation=True, return_tensors="pt")

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=260,
        min_length=140,
        early_stopping=True,
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ------------------------------------------------------
# ROUGE Score Table
# ------------------------------------------------------
def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def format_rouge(scores):
    data = {
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "Precision": [
            scores["rouge1"].precision,
            scores["rouge2"].precision,
            scores["rougeL"].precision,
        ],
        "Recall": [
            scores["rouge1"].recall,
            scores["rouge2"].recall,
            scores["rougeL"].recall,
        ],
        "F1 Score": [
            scores["rouge1"].fmeasure,
            scores["rouge2"].fmeasure,
            scores["rougeL"].fmeasure,
        ],
    }
    return pd.DataFrame(data)

# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.set_page_config(page_title="Hybrid Summarizer (LexRank + BART)", layout="wide")
st.title("Text Summarizer")

text = st.text_area("Enter text to summarize:", height=280)

tone = st.selectbox(
    "Choose Tone:",
    ["Neutral", "Formal", "News-like", "Academic"]
)

if st.button("Generate Summary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # Count input words
        input_word_count = len(text.split())

        # 1. Preprocess
        original_sentences, _ = preprocess_text(text)

        # 2. Extractive Summary
        ext = extractive_summary(original_sentences)

        # 3. Abstractive Summary (Final)
        final_summary = abstractive_summary(ext, tone)

        # Count summary words
        summary_word_count = len(final_summary.split())

        # 4. ROUGE Evaluation
        rouge_scores = compute_rouge(text, final_summary)
        df = format_rouge(rouge_scores)

        # ------------------------------------------------------
        # SIDE-BY-SIDE VIEW
        # ------------------------------------------------------
        st.subheader("Side-by-Side Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Original Text")
            st.write(text)

        with col2:
            st.markdown("Summary")
            st.write(final_summary)

        # ------------------------------------------------------
        # METRICS
        # ------------------------------------------------------
        st.subheader("Word Count")
        st.write(f"**Words in Original Text:** {input_word_count}")
        st.write(f"**Words in Summary:** {summary_word_count}")

        st.subheader("ROUGE Evaluation")

        numeric_cols = ["Precision", "Recall", "F1 Score"]

        st.dataframe(df.style.format({col: "{:.3f}" for col in numeric_cols}))
