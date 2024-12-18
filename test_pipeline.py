from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
res = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print(res)