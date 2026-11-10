
from transformers import pipeline


def load_model():
    classifier = pipeline(
        task="sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    return classifier


def predict_sentiment(classifier, texts):
    predictions = classifier(texts, batch_size=8)
    return predictions


def display_results(texts, predictions):
    for text, result in zip(texts, predictions):
        label = result["label"]
        confidence = result["score"]

        print("=" * 60)
        print(f"Text       : {text}")
        print(f"Sentiment  : {label}")
        print(f"Confidence : {confidence:.2%}")

        if confidence >= 0.90:
            print("Interpretation: High model confidence")
        elif confidence >= 0.70:
            print("Interpretation: Moderate model confidence")
        else:
            print("Interpretation: Lower model confidence")

    print("=" * 60)


def main():
    print("Loading Transformer model...")

    classifier = load_model()

    print("Model loaded successfully!\n")

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

    predictions = predict_sentiment(classifier, examples)

    display_results(examples, predictions)


if __name__ == "__main__":
    main()
