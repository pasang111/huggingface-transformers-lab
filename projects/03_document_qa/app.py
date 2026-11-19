
"""Project 3: Question Answering System."""

import re
from collections import Counter


def tokenize(text):
    """Convert text into lowercase words."""
    return re.findall(r"\b\w+\b", text.lower())


def split_sentences(context):
    """Split context into individual sentences."""
    return re.split(r"(?<=[.!?])\s+", context.strip())


def find_answer(context, question):
    """Find the sentence most relevant to the question."""
    sentences = split_sentences(context)
    question_words = set(tokenize(question))

    if not question_words:
        return "Please enter a valid question."

    best_sentence = None
    best_score = 0

    for sentence in sentences:
        sentence_words = tokenize(sentence)
        word_counts = Counter(sentence_words)

        score = sum(
            word_counts[word]
            for word in question_words
        )

        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_sentence is None:
        return "Sorry, I couldn't find an answer in the supplied context."

    return best_sentence


def main():
    print("=" * 45)
    print("       QUESTION ANSWERING SYSTEM")
    print("=" * 45)

    context = input("\nEnter the context:\n")

    if not context.strip():
        print("Error: Context cannot be empty.")
        return

    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")

        if question.lower().strip() == "exit":
            print("Thank you for using the Question Answering System!")
            break

        answer = find_answer(context, question)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
