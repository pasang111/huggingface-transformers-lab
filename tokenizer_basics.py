from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased"
)

text = "I love learning Python!!"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
decode_text = tokenizer.decode(token_ids)

print("Original Text: ", text)
print("Tokens: ", tokens)
print("Token Ids: ", token_ids)
print("Decoded Text: ", decode_text)