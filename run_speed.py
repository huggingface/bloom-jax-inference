import numpy as np
import jax
import time

from bloom_inference import FlaxBloomForCausalLM, BloomConfig
from transformers import AutoTokenizer

from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate

num_devices = jax.local_device_count()

max_length = 256
model = FlaxBloomForCausalLM.from_pretrained("sanchit-gandhi/bloom-760m-scan", use_scan=True, do_sample=True, max_length=max_length)
params = model.params
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-350m")

prompt = "hello there this is a longer sentence"

rng = jax.random.PRNGKey(0)
rngs = jax.random.split(rng, num_devices)

def generate(params, inputs, rng):
    output_ids = model.generate(**inputs, prng_key=rng, params=params).sequences
    return output_ids

p_generate = jax.pmap(generate, "batch")
p_params = replicate(model.params)

def run_generate(input_str):
    inputs = tokenizer([input_str for i in range(num_devices)], return_tensors="jax", padding="max_length", truncation=True, max_length=32)
    p_inputs = shard(inputs.data)
    output_ids = p_generate(p_params, p_inputs, rngs)
    output_strings = tokenizer.batch_decode(output_ids.reshape(-1, max_length), skip_special_tokens=True)
    return output_strings

print("generated text", run_generate(prompt))

print("----------------------------")
start_time = time.time()
out = run_generate(prompt)
print(len(out))
print(time.time() - start_time)
print("----------------------------")
start_time = time.time()
out = run_generate(prompt)
print(len(out))
print(time.time() - start_time)
print("----------------------------")
start_time = time.time()
out = run_generate(prompt)
print(len(out))
print(time.time() - start_time)
