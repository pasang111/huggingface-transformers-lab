
from transformers import pipeline
import torch


def load_model():
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        task="sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

    return classifier


def predict_sentiment(classifier, texts):
    if not texts:
        return []

    if isinstance(texts, str):
        texts = [texts]

    predictions = classifier(
        texts,
        batch_size=8,
        truncation=True
    )

    return predictions


# CMR: Confidence-Based Model Reliability
def calculate_cmr(predictions):
    if not predictions:
        return 0, 0, 0

    total = len(predictions)

    high_confidence = sum(
        1 for p in predictions if p["score"] >= 0.90
    )

    moderate_confidence = sum(
        1 for p in predictions if 0.70 <= p["score"] < 0.90
    )

    low_confidence = sum(
        1 for p in predictions if p["score"] < 0.70
    )

    cmr = (
        sum(p["score"] for p in predictions) / total
    ) * 100

    return cmr, high_confidence, moderate_confidence, low_confidence


def display_results(texts, predictions):
    print("\nSENTIMENT ANALYSIS RESULTS")
    print("=" * 60)

    for text, result in zip(texts, predictions):

        label = result["label"]
        confidence = result["score"]

        if confidence >= 0.90:
            interpretation = "High confidence"
        elif confidence >= 0.70:
            interpretation = "Moderate confidence"
        else:
            interpretation = "Low confidence"

        print(f"\nText       : {text}")
        print(f"Sentiment  : {label}")
        print(f"Confidence : {confidence:.2%}")
        print(f"Analysis   : {interpretation}")

    # Calculate CMR
    cmr, high, moderate, low = calculate_cmr(predictions)

    print("\n" + "=" * 60)
    print("CMR (CONFIDENCE-BASED MODEL RELIABILITY)")
    print("=" * 60)

    print(f"Overall CMR       : {cmr:.2f}%")
    print(f"High Confidence   : {high}")
    print(f"Moderate Confidence: {moderate}")
    print(f"Low Confidence    : {low}")

    print("=" * 60)


def main():
    examples = [
        "I really enjoyed learning about transformers today.",
        "This was confusing and frustrating.",
        "The package arrived yesterday.",
        "The movie was absolutely fantastic!",
        "I am disappointed with the customer service.",
        "The weather is nice today.",
        "Artificial intelligence is changing the world.",
        "I hate waiting for hours."
    ]

    try:
        print("Loading sentiment analysis model...")

        classifier = load_model()

        print("Model loaded successfully!")

        predictions = predict_sentiment(classifier, examples)

        display_results(examples, predictions)

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
