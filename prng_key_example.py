import time
from bloom_inference.generator import head_print, Generator

head_print("Initialising model...")
start = time.time()
generator = Generator(num_mp_partitions=4, max_len=16, max_input_len=8)
head_print(f"Model initialised in {time.time() - start:.06}s")

head_print("Loading weights...")
start = time.time()
generator.load_model_and_params()
head_print(f"Weights loaded in {time.time() - start:.06}s")

head_print("Starting compile...")
# benchmark compile step
start = time.time()
head_print(generator.generate(2*['Recipe for coconut '], do_sample=True))
head_print(f"Generations completed in {time.time() - start:.06}s")

# benchmark generate
head_print("Starting generate...")
start = time.time()
head_print(generator.generate(2*['Recipe for coconut '], do_sample=True))
head_print(f"Generations completed in {time.time() - start:.06}s")