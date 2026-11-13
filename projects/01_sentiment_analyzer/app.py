
from transformers import pipeline
import torch


# Load the pre-trained sentiment analysis model
def load_model():
    # Automatically select GPU if available, otherwise use CPU
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        task="sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

    return classifier


# Predict sentiment for one or multiple text inputs
def predict_sentiment(classifier, texts):

    # Return an empty list if no input is provided
    if texts is None or texts == []:
        return []

    # Convert a single string into a list
    if isinstance(texts, str):
        texts = [texts]

    # Validate input data
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("All inputs must be non-empty strings.")

    # Generate sentiment predictions in batches
    predictions = classifier(
        texts,
        batch_size=8,
        truncation=True
    )

    return predictions


# Calculate CMR (Confidence-Based Model Reliability)
def calculate_cmr(predictions):

    # Handle empty prediction results
    if not predictions:
        return 0, 0, 0, 0

    total = len(predictions)

    # Count predictions based on confidence levels
    high_confidence = sum(
        1 for p in predictions if p["score"] >= 0.90
    )

    moderate_confidence = sum(
        1 for p in predictions if 0.70 <= p["score"] < 0.90
    )

    low_confidence = sum(
        1 for p in predictions if p["score"] < 0.70
    )

    # Calculate average confidence as a percentage
    cmr = (
        sum(p["score"] for p in predictions) / total
    ) * 100

    return cmr, high_confidence, moderate_confidence, low_confidence


# Display individual predictions and overall CMR results
def display_results(texts, predictions):

    print("\n" + "=" * 65)
    print("             SENTIMENT ANALYSIS RESULTS")
    print("=" * 65)

    # Display sentiment and confidence for each input
    for index, (text, result) in enumerate(zip(texts, predictions), 1):

        label = result["label"]
        confidence = result["score"]

        # Classify the prediction confidence
        if confidence >= 0.90:
            interpretation = "High Confidence"
        elif confidence >= 0.70:
            interpretation = "Moderate Confidence"
        else:
            interpretation = "Low Confidence"

        print(f"\nText #{index}   : {text}")
        print(f"Sentiment     : {label}")
        print(f"Confidence    : {confidence:.2%}")
        print(f"Reliability   : {interpretation}")
        print("-" * 65)

    # Calculate overall CMR statistics
    cmr, high, moderate, low = calculate_cmr(predictions)

    # Display the CMR summary
    print("\n" + "=" * 65)
    print("       CMR (CONFIDENCE-BASED MODEL RELIABILITY)")
    print("=" * 65)

    print(f"Overall CMR          : {cmr:.2f}%")
    print(f"Total Predictions    : {len(predictions)}")
    print(f"High Confidence      : {high}")
    print(f"Moderate Confidence  : {moderate}")
    print(f"Low Confidence       : {low}")

    print("=" * 65)


# Main function to execute sentiment analysis
def main():

    # Sample text inputs for testing
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
        # Load the sentiment analysis model
        print("\nLoading sentiment analysis model...")

        classifier = load_model()

        print("Model loaded successfully!")

        # Generate predictions
        predictions = predict_sentiment(classifier, examples)

        # Display prediction results and CMR
        display_results(examples, predictions)

    except ValueError as error:
        print(f"\nInput Error: {error}")

    except Exception as error:
        print(f"\nUnexpected Error: {error}")


# Run the program
if __name__ == "__main__":
    main()
