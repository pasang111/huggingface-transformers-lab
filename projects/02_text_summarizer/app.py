
"""
Project 2: AI Text Summarization using Hugging Face.

Features:
- Abstractive text summarization
- Interactive CLI
- Input validation
- Summary statistics
- Configurable summary length
- Error handling
"""

from transformers import pipeline
import sys


MODEL_NAME = "facebook/bart-large-cnn"


def load_summarizer():
    """Load the Hugging Face summarization model."""
    print("\nLoading AI summarization model...")

    try:
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME
        )
        print("Model loaded successfully!")
        return summarizer

    except Exception as error:
        print(f"\nError loading model: {error}")
        print("Check your internet connection and dependencies.")
        sys.exit(1)


def get_user_input():
    """Get text input from the user."""
    print("\nEnter the text you want to summarize.")
    print("(Type END on a new line when finished.)\n")

    lines = []

    while True:
        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    text = "\n".join(lines).strip()

    if not text:
        print("Error: Input cannot be empty.")
        return None

    return text


def summarize_text(summarizer, text):
    """Generate a summary from the provided text."""
    try:
        word_count = len(text.split())

        if word_count < 30:
            print("\nText is already relatively short.")
            print("Please provide at least 30 words.")
            return

        print("\nGenerating summary... Please wait.")

        # Keep input within the model's token limit.
        max_input_tokens = 1024

        inputs = summarizer.tokenizer(
            text,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors=None
        )

        truncated_text = summarizer.tokenizer.decode(
            inputs["input_ids"],
            skip_special_tokens=True
        )

        summary = summarizer(
            truncated_text,
            max_length=130,
            min_length=30,
            do_sample=False,
            truncation=True
        )

        summary_text = summary[0]["summary_text"]

        # Calculate statistics
        summary_words = len(summary_text.split())

        compression = (
            (1 - summary_words / word_count) * 100
        )

        print("\n" + "=" * 50)
        print("                 GENERATED SUMMARY")
        print("=" * 50)

        print(f"\n{summary_text}")

        print("\n" + "=" * 50)
        print("                 SUMMARY STATISTICS")
        print("=" * 50)

        print(f"Original words : {word_count}")
        print(f"Summary words  : {summary_words}")
        print(f"Compression    : {compression:.2f}%")

        print("\nSummary generated successfully!")

    except Exception as error:
        print(f"\nSummarization failed: {error}")


def main():
    """Main application entry point."""

    print("\n" + "=" * 50)
    print("       AI TEXT SUMMARIZATION APPLICATION")
    print("=" * 50)

    print(f"\nModel: {MODEL_NAME}")

    summarizer = load_summarizer()

    while True:
        text = get_user_input()

        if text:
            summarize_text(summarizer, text)

        choice = input(
            "\nWould you like to summarize another text? (y/n): "
        ).strip().lower()

        if choice not in ("y", "yes"):
            print("\nThank you for using AI Text Summarization!")
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
