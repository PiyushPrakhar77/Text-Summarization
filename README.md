Hybrid Text Summarizer (LexRank + BART)

A web-based text summarization system combining extractive and abstractive NLP techniques to produce high-quality summaries for long-form text. Built with Streamlit, NLTK, LexRank, and BART-Large CNN.

Features

Hybrid Summarization Pipeline using LexRank (extractive) + BART (abstractive) for more accurate and context-aware summaries.

NLP Preprocessing with tokenization, stopword removal, lemmatization, and text normalization using NLTK.

Tone Control option (Neutral, Formal, News-like, Academic) for generating summaries in the desired writing style.

ROUGE Evaluation Metrics (ROUGE-1, ROUGE-2, ROUGE-L) displayed in a structured table.

Interactive UI built with Streamlit featuring side-by-side original and summarized text comparison.

Word Count Analysis for both input text and generated summary.

Tech Stack

  Python 3.10+
  
  Streamlit
  
  NLTK
  
  LexRank
  
  Transformers (HuggingFace)
  
  BART-Large CNN
  
  ROUGE Score (rouge-score library)
  
  Pandas

📦 Installation
git clone https://github.com/<your-username>/hybrid-text-summarizer.git
cd hybrid-text-summarizer
pip install -r requirements.txt


Make sure to install PyTorch based on your system configuration:
https://pytorch.org/get-started/locally/

▶️ Run the App
streamlit run app.py

📚 How It Works

User enters the text to summarize.

Text undergoes NLP preprocessing.

LexRank generates an extractive base summary.

BART-Large CNN converts it into an abstractive summary.

ROUGE metrics evaluate summary quality.

Final output is displayed in a clean UI with comparison and word count.

📊 Sample Output

Extractive summary (LexRank)

Abstractive summary (BART)

ROUGE metrics table

Word count stats

Side-by-side viewer

📝 Future Improvements

Add support for multilingual summarization

Integrate T5 / Flan-T5 models

Option to upload documents (PDF, TXT)

Improve UI with theme customization

Add GPT-based fast summarization mode
