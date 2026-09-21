import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


DATA_FILE = Path(__file__).with_name("reviews.csv")


def preprocess_text(text: str) -> str:
    """Clean text and tokenize it into a normalized space-separated string."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = re.findall(r"[a-z]+", text)
    return " ".join(tokens)


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)

    required = {"text", "sentiment"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain 'text' and 'sentiment' columns.")

    data = data.dropna(subset=["text", "sentiment"]).copy()
    data["clean_text"] = data["text"].apply(preprocess_text)
    return data


def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["sentiment"],
        test_size=0.25,
        random_state=42,
        stratify=data["sentiment"],
    )

    # Convert text to numerical TF-IDF features.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression classifier.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    print("\n=== Model Evaluation ===")
    print(f"Accuracy : {accuracy:.2%}")
    print(f"F1 Score : {f1:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return vectorizer, model


def predict_sentiment(text, vectorizer, model):
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned])
    return model.predict(features)[0]


def main():
    print("=" * 55)
    print("        SENTIMENT ANALYSIS TOOL")
    print("      TF-IDF + Logistic Regression")
    print("=" * 55)

    data = load_data()
    print(f"\nLoaded {len(data)} labeled reviews.")

    vectorizer, model = train_model(data)

    print("\n=== Sentiment Prediction CLI ===")
    print("Type a review and press Enter.")
    print("Type 'exit' to close the program.\n")

    while True:
        user_text = input("Enter text: ").strip()

        if user_text.lower() == "exit":
            print("Thank you for using the Sentiment Analysis Tool!")
            break

        if not user_text:
            print("Please enter some text.\n")
            continue

        sentiment = predict_sentiment(user_text, vectorizer, model)
        print(f"Predicted sentiment: {sentiment.upper()}\n")


if __name__ == "__main__":
    main()
