# Sentiment Analysis Tool

## Syntecxhub Artificial Intelligence Internship – Week 2

A small Natural Language Processing (NLP) project that classifies text reviews as **positive** or **negative**.

### Features
- Loads labeled review data from `reviews.csv`
- Cleans and tokenizes text
- Converts text into numerical features using **TF-IDF**
- Trains a **Logistic Regression** classifier
- Evaluates the model using **Accuracy** and **F1 Score**
- Provides an interactive command-line interface (CLI)
- Predicts the sentiment of new user-entered text

### Project Structure

```text
sentiment-analysis-tool/
├── reviews.csv
├── sentiment_analysis.py
├── requirements.txt
└── README.md
```

### Requirements

- Python 3.9+
- pandas
- scikit-learn

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python sentiment_analysis.py
```

### Methodology

1. Load labeled review text.
2. Convert text to lowercase, remove URLs and non-letter characters, and tokenize it.
3. Use `TfidfVectorizer` to convert text into numerical features.
4. Split the dataset into training and testing sets.
5. Train a Logistic Regression classifier.
6. Measure Accuracy and weighted F1 Score.
7. Accept new text through the CLI and predict its sentiment.

### Example

```text
Enter text: I really enjoyed this product and the quality is excellent.
Predicted sentiment: POSITIVE

Enter text: The service was terrible and very disappointing.
Predicted sentiment: NEGATIVE
```

### Task Alignment

This project implements the requirements listed in the Syntecxhub Week 2 AI task:
- labeled text data
- text preprocessing
- CountVectorizer or TF-IDF
- Naive Bayes or Logistic Regression
- accuracy and F1 evaluation
- CLI sentiment prediction
