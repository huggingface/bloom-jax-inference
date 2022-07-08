import warnings

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax.experimental import PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache as cc

from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState

from bloom_inference.modeling_bloom import FlaxBloomForCausalLM
from transformers import AutoTokenizer

cc.initialize_cache("~/jax_cache")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ResourceWarning)

if jax.process_index() == 0:
    warnings.filterwarnings("default")

# print but only on the first node
def head_print(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)

# 2D parameter and activation partitioning
logical_axis_rules_full = [
            ('batch', 'data'),
            ('mlp', 'model'),
            ('heads', 'model'),
            ('vocab', 'model'),
            # shard both activations and weight matrices on the remaining available axis
            ('embed', 'model'),
            ('embed', 'data'),
            ('kv', None),
            ('joined_kv', None),
            ('relpos_buckets', None),
            ('abspos_buckets', None),
            ('length', None),
            ('layers', None),
            ('stack', None),
            ('mlp_activations', None),
        ]


class Generator:
    def __init__(self, num_mp_partitions, ckpt="bigscience/bloom-6b3", max_len=64, num_beams=1, do_sample=False):
        # create a mesh and bind names to mesh axes
        self.num_mp_partitions = num_mp_partitions

        self.ckpt = ckpt

        if ckpt.split("-")[-1] == "scan":
            self.use_scan = True
        else:
            self.use_scan = False

    def init_fn(self):
        input_shape = (1, 1)
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        rng = jax.random.PRNGKey(0)
        return self.model.module.init(rng, input_ids, attention_mask, return_dict=False)

    def load_model_and_params(self):
        # TODO loading params should be done in a thread
        flax_ckpt = "sanchit-gandhi/bloom-350m-scan-t5x"
        tok_ckpt = "bigscience/bloom-350m"

        model, self.params = FlaxBloomForCausalLM.from_pretrained(
            flax_ckpt,
            _do_init=False,
            use_scan=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(tok_ckpt, use_fast=False)

        # setup for generation
        tokenizer.padding_sided = "left"
        model.config.max_length = 64
        model.config.num_beams = 1
        model.config.do_sample = False

        self.model = model
        self.tokenizer = tokenizer

        # Axis names metadata
        param_axes = jax.eval_shape(self.init_fn)["params_axes"]

        # create InferenceState, since the partitioner expects it.
        state = InferenceState(
            step=jnp.array(0),
            params=freeze(model.params_shape_tree),
            params_axes=freeze(param_axes),
            flax_mutables=None,
            flax_mutables_axes=param_axes,
        )

        # TODO: fix num_mp_partitions arg - hard code to 8 for now (self.num_mp_partitions = 32)
        self.partitioner = PjitPartitioner(8, logical_axis_rules=logical_axis_rules_full)

        mesh_axes = self.partitioner.get_mesh_axes(state)
        self.params_spec = mesh_axes.params

        self.p_shard_params = self.partitioner.partition(self.model.to_bf16, (self.params_spec,), self.params_spec)

        def generate(params, input_ids, attention_mask):
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
            return output_ids

        self.p_generate = self.partitioner.partition(
            generate,
            in_axis_resources=(self.params_spec, P("data"), P("data")),
            out_axis_resources=P("data")
        )

    def shard_params(self):
        # This will auto-magically run in mesh context
        self.params = self.p_shard_params(freeze(self.params))

    def generate(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="jax", padding="max_length", truncation=True, max_length=64) # BS = 4
        # This will auto-magically run in mesh context
        gen_ids = self.p_generate(freeze(self.params), inputs["input_ids"], inputs["attention_mask"])

        generated_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return generated_text
