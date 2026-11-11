from transformers import pipeline  # Import pipeline from transformers library


def load_model():  # Function to load the sentiment analysis model
    classifier = pipeline(  # Create a sentiment analysis pipeline
        task="sentiment-analysis",  # Specify the task as sentiment analysis
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"  # Select the pretrained model
    )
    return classifier  # Return the loaded model


def predict_sentiment(classifier, texts):  # Function to predict sentiment
    predictions = classifier(texts, batch_size=8)  # Predict sentiment for all texts in batches of 8
    return predictions  # Return the prediction results


def display_results(texts, predictions):  # Function to display prediction results
    for text, result in zip(texts, predictions):  # Loop through texts and their predictions
        label = result["label"]  # Get the sentiment label (POSITIVE or NEGATIVE)
        confidence = result["score"]  # Get the confidence score

        print("=" * 60)  # Print a separator line
        print(f"Text       : {text}")  # Display the input text
        print(f"Sentiment  : {label}")  # Display the predicted sentiment
        print(f"Confidence : {confidence:.2%}")  # Display confidence as a percentage

        if confidence >= 0.90:  # Check if confidence is 90% or higher
            print("Interpretation: High model confidence")  # Display high confidence
        elif confidence >= 0.70:  # Check if confidence is 70% or higher
            print("Interpretation: Moderate model confidence")  # Display moderate confidence
        else:  # If confidence is below 70%
            print("Interpretation: Lower model confidence")  # Display lower confidence

    print("=" * 60)  # Print the final separator line


def main():  # Main function of the program
    print("Loading Transformer model...")  # Display loading message

    classifier = load_model()  # Load the sentiment analysis model

    print("Model loaded successfully!\n")  # Display success message

    examples = [  # Create a list of example sentences
        "I really enjoyed learning about transformers today.",  # Positive sentence
        "This was confusing and frustrating.",  # Negative sentence
        "The package arrived yesterday.",  # Neutral-sounding sentence
        "The movie was absolutely fantastic!",  # Positive sentence
        "I am disappointed with the customer service.",  # Negative sentence
        "The weather is nice today.",  # Positive sentence
        "Artificial intelligence is changing the world.",  # Neutral or positive-sounding sentence
        "I hate waiting for hours."  # Negative sentence
    ]

    predictions = predict_sentiment(classifier, examples)  # Predict sentiment for all examples

    display_results(examples, predictions)  # Display all prediction results


if __name__ == "__main__":  # Check if this file is running directly
    main()  # Call the main function to execute the program
