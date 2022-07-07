import re

from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import PartitionSpec as P


# Sentinels
_unmatched = object()

# For specifying empty leaf dict `***REMOVED******REMOVED***`
empty_dict = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace


def _get_partition_rules():
    return [
        # Embeddings
        (("word_embeddings", "embedding"), P(None, "mp")),
        # Attention
        ((r"self_attention", "query_key_value", "kernel"), P("mp", None)),
        ((r"self_attention", "query_key_value", "bias"), P(None)),
        ((r"self_attention", "dense", "kernel"), P(None, "mp")),
        ((r"self_attention", "dense", "kernel"), P("mp")),
        # FFN
        ((r"mlp", "dense_4h_to_h", "kernel"), P("mp", None)),
        ((r"mlp", "dense_4h_to_h", "bias"), P(None)),
        ((r"mlp", "dense_h_to_4h", "kernel"), P(None, "mp")),
        ((r"mlp", "dense_h_to_4h", "bias"), P("mp")),
        # layer norms
        (("(bias|scale)",), None),
        # projection
        # (("lm_head", "kernel"), P(None, "mp")),
    ]


def set_partitions(in_dict):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = ***REMOVED***k: _unmatched for k in flatten_dict(in_dict)***REMOVED***
    result = ***REMOVED***k: replace(k, v) for k, v in initd.items()***REMOVED***
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))