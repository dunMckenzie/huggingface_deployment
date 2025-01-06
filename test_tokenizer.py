from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("gem_marketing")
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

