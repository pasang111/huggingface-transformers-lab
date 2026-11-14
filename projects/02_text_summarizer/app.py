"""Project 2: Text Summarization using Hugging Face."""

from transformers import pipeline


def main():
    # Load the summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    print("=== Text Summarization App ===")

    # Get text from the user
    text = input("\nEnter the text you want to summarize:\n")

    if not text.strip():
        print("Error: Please enter some text.")
        return

    # Generate summary
    print("\nGenerating summary...")

    summary = summarizer(
        text,
        max_length=130,
        min_length=30,
        do_sample=False
    )

    # Display result
    print("\n=== Summary ===")
    print(summary[0]["summary_text"])


if __name__ == "__main__":
    main()
