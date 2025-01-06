from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# Input text
input_text = "Write me a poem about Machine Learning."

# Tokenize the input
input_ids = tokenizer(input_text, return_tensors="pt")

# Generate output
outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
