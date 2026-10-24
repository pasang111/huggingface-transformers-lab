def test_example_sentences_are_nonempty():
    examples = [
        "I really enjoyed learning about transformers today.",
        "This was confusing and frustrating.",
    ]
    assert all(isinstance(text, str) and text.strip() for text in examples)
