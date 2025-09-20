# NLP Movie Review Sentiment Analyzer

This project builds and compares three different machine learning models to classify movie reviews as either positive or negative. The entire workflow, from data cleaning to final model evaluation and an interactive demo, is documented in a series of Jupyter Notebooks.

![Sentiment Analyzer Demo](GIF_NLP)

## Project Goal
The objective of this project was to apply the complete lifecycle of a Natural Language Processing project to a real-world problem. The goal was to develop a high-performance sentiment analysis model, starting with a simple baseline and progressively iterating with more advanced architectures to achieve a high degree of precision and recall.

## Dataset
This project uses the **IMDB Dataset of 50K Movie Reviews**, a public dataset available on Kaggle. It is perfectly balanced with 25,000 positive and 25,000 negative reviews, making it an excellent benchmark for binary classification.

## Methodology
The project followed a systematic, multi-step approach, with each step documented in a separate notebook:

1.  **Data Exploration and Cleaning:** The raw data was loaded, explored, and found to contain HTML tags and other noise. A robust preprocessing pipeline was built using `spaCy` to clean and lemmatize the text while strategically preserving all stop words for the N-gram model.

2.  **Baseline Model:** A classic NLP model was built using a `TfidfVectorizer` (with 1, 2, and 3-grams) and a `LogisticRegression` classifier. This provided a strong baseline F1-score to beat.

3.  **Advanced Neural Model:** A simple feed-forward neural network (`MLPClassifier`) was trained using pre-trained 300-dimension word vectors from `spaCy`'s `en_core_web_md` model.

4.  **State-of-the-Art Model:** A pre-trained **Transformer** model (`DistilBERT` from the Hugging Face library) was implemented to leverage its deep contextual understanding of language.

## Final Results
The performance of all three models was evaluated on an unseen test set using the F1-Score as the primary metric.

| Model                       | F1-Score (Weighted Avg) |
| --------------------------- | ----------------------- |
| Baseline (TF-IDF + N-grams) | 0.71                    |
| Advanced (Word Vectors + NN)  | 0.70                    |
| **State-of-the-Art (Transformer)** | **0.89** |

The Transformer-based model was the clear winner, demonstrating the effectiveness of modern, context-aware NLP architectures.

## Interactive Demo
An interactive demo was built using `Gradio` to allow for real-time sentiment analysis of custom text. To run it, execute the final cells in the `4_Transformer_Model.ipynb` notebook.

## How to Run
1.  Clone the repository: `git clone https://github.com/YourUsername/Your-Repo-Name.git`
2.  Install all necessary libraries: `pip install -r requirements.txt`
3.  The notebooks are numbered and are designed to be run in order.
