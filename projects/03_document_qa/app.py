
"""Project 3: Question Answering System."""

import re
from collections import Counter


def tokenize(text):
    """Convert text into normalized lowercase words."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def split_sentences(context):
    """Split context into individual sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", context.strip())
        if sentence.strip()
    ]


def find_answer(context, question):
    """Find the sentence that best matches the question."""
    if not context or not context.strip():
        return "Error: Context cannot be empty."

    if not question or not question.strip():
        return "Please enter a valid question."

    sentences = split_sentences(context)
    question_words = set(tokenize(question))

    if not question_words:
        return "Please enter a valid question."

    # Common words that provide little meaning.
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were",
        "do", "does", "did", "what", "when", "where",
        "who", "why", "how", "which", "to", "of",
        "in", "on", "at", "and", "or", "for", "with"
    }

    keywords = question_words - stop_words

    if not keywords:
        keywords = question_words

    best_sentence = None
    best_score = 0

    for sentence in sentences:
        sentence_words = tokenize(sentence)
        word_counts = Counter(sentence_words)

        # Calculate relevance using keyword frequency.
        matched_words = keywords.intersection(word_counts)

        score = sum(
            word_counts[word]
            for word in matched_words
        )

        # Prefer sentences that match more unique keywords.
        coverage = len(matched_words) / len(keywords)

        # Combine keyword frequency and coverage.
        final_score = score + coverage

        if final_score > best_score:
            best_score = final_score
            best_sentence = sentence

    if best_sentence is None or best_score == 0:
        return "Sorry, I couldn't find an answer in the supplied context."

    return best_sentence


def main():
    """Run the interactive question-answering system."""
    print("=" * 50)
    print("          QUESTION ANSWERING SYSTEM")
    print("=" * 50)

    context = input("\nEnter the context:\n").strip()

    if not context:
        print("\nError: Context cannot be empty.")
        return

    print("\nContext successfully loaded!")
    print("You can now ask questions about the given context.")

    while True:
        question = input(
            "\nAsk a question (or type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":
            print("\nThank you for using the Question Answering System!")
            break

        answer = find_answer(context, question)

        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
