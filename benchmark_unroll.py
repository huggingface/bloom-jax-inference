import time
from bloom_inference.generator import head_print, Generator
from transformers import BloomConfig

config = BloomConfig.from_pretrained("bigscience/bloom")

batch_size = 2

head_print("max_inp = 8, max_new = 8, bs = 2")

num_unroll = 1

generator = Generator(num_mp_partitions=4, max_len=16, max_input_len=8, unroll=num_unroll)

generator.load_model_and_params()

# compile step
generator.generate(batch_size*['Recipe for a quick coconut pasta:'], do_sample=True)

# benchmark generate
start = time.time()
generator.generate(batch_size*['Recipe for a quick coconut pasta:'], do_sample=True)
head_print(f"Unroll {num_unroll}: {time.time() - start:.06}s")

del generator


num_unroll = 2

generator = Generator(num_mp_partitions=4, max_len=16, max_input_len=8, unroll=num_unroll)

generator.load_model_and_params()

# compile step
generator.generate(batch_size*['Recipe for a quick coconut pasta:'], do_sample=True)

# benchmark generate
start = time.time()
generator.generate(batch_size*['Recipe for a quick coconut pasta:'], do_sample=True)
head_print(f"Unroll {num_unroll}: {time.time() - start:.06}s")

del generator
