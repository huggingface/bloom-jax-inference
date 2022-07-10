import numpy as np

from bloom_inference import FlaxBloomForCausalLM, BloomConfig
from transformers import AutoTokenizer

vocab_size = 64
max_length = 15
config = BloomConfig(n_layer=2, vocab_size=vocab_size, hidden_size=32, n_head=2, pad_token_id=0)

model = FlaxBloomForCausalLM(config)
#model = FlaxBloomForCausalLM.from_pretrained("bloom-jax-dummy", local_files_only=True, max_length=32)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-350m", use_fast=False, local_files_only=True)

prompts = ["Hello", "hello there this is a longer sentence"]

tokenizer.padding_side = "left"
inputs = tokenizer(prompts, return_tensors="jax", padding="max_length", truncation=True, max_length=8)
inputs["input_ids"] = inputs["input_ids"] % vocab_size

print(40 * "-")
expected_ids = """
Ids [[ 3  3  3  3  3  3  3  3  3  3  3  3  3  3  3 22 22 22 22 22]
 [ 3  3  3  3  3  3  3  3  3 11 30 31 56 11 27 52 52 52 52 52]]
"""
print("EXPECTED ids:", expected_ids)
gen_ids = model.generate(**inputs, trace=False).sequences
print("Ids", gen_ids)
expected_str = "generated text ['33333', '(;<U(8QQQQQ'"
print(40 * "-")
print("EXPECTED str:", expected_str)
generated_text = tokenizer.batch_decode(np.asarray(gen_ids), skip_special_tokens=True)
print("generated text", generated_text)
