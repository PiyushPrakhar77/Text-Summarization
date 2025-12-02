Hybrid Text Summarizer — LexRank + BART (NLP Project)

An advanced text-summarization system that blends extractive and abstractive AI models to generate clear, coherent, and high-quality summaries. Designed with a modern Streamlit interface and powered by state-of-the-art NLP techniques.

🚀 Key Features

🔹 Hybrid AI Pipeline: LexRank for extractive summarization + BART-Large CNN for abstractive refinement.

🔹 Smarter Preprocessing: Tokenization, lemmatization, stopword removal, and text normalization using NLTK.

🔹 Tone Customization: Choose from Neutral, Formal, News-like, or Academic writing styles.

🔹 ROUGE Evaluation: Built-in ROUGE-1, ROUGE-2, and ROUGE-L scoring for quality measurement.

🔹 Interactive UI: Clean Streamlit interface with side-by-side original vs. summary comparison.

🔹 Word Count Insights: Automatic word count for both input and generated summary.

🧠 Tech Stack
Category	Tools
Frontend / UI	Streamlit
NLP	NLTK, LexRank, HuggingFace Transformers
Model	BART-Large CNN
Evaluation	ROUGE Scorer
Language	Python
Data	Pandas
📦 Installation

Clone the repository:

git clone https://github.com/PiyushPrakhar77/Text-Summarization.git
cd Text-Summarization


Install dependencies:

pip install -r requirements.txt


Install PyTorch (based on your system):
https://pytorch.org/get-started/locally/

▶️ Run the Application
streamlit run app.py

🔍 How the System Works

User inputs long text into the UI.

The system preprocesses text using NLTK.

LexRank extracts core sentences.

BART-Large rewrites them into an abstractive, human-like summary.

ROUGE metrics measure accuracy and relevance.

Output appears in a polished side-by-side viewer.

📊 Output Includes

Extractive summary

Abstractive summary

ROUGE precision, recall, and F1 scores

Word count stats

Comparison layout

🌱 Future Enhancements

Support for multiple languages

Document upload (PDF, DOCX)

Integration of T5 and GPT-based summarizers

Improved UI themes

API endpoint for developers
