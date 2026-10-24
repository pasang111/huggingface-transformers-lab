"""Project 1: classify text sentiment using a pretrained Transformer."""
from transformers import pipeline

def main():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    examples = [
        "I really enjoyed learning about transformers today.",
        "This was confusing and frustrating.",
        "The package arrived yesterday.",
    ]
    for text, result in zip(examples, classifier(examples)):
        print(f"Text: {text}")
        print(f"Label: {result['label']}")
        print(f"Score: {result['score']:.3f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
