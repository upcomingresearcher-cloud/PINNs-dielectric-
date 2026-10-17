"""Fully connected tanh networks (protocol Sec. 5): Glorot-normal weights, zero biases.

Initial weights are ALWAYS drawn in float32 and then cast to the run dtype, so a float64 run starts from exactly
the same values as the float32 run with the same seed (Intervention C requires "everything else identical").
Bug fix 2026-10-16 (change log v1.6.0): previously jax.random.normal was called with the run dtype, which gives a
different draw in float64. float32 initialisations are bit-identical before and after the fix."""
import jax, jax.numpy as jnp


def init_mlp(key, d_in, width, depth, dtype):
    """Fully connected tanh MLP with Glorot-normal weights (drawn in float32, cast to dtype) and zero biases."""
    sizes = [d_in] + [width] * depth + [1]
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        w32 = jax.random.normal(k, (fan_in, fan_out), jnp.float32) * std.astype(jnp.float32)
        params.append((w32.astype(dtype), jnp.zeros((fan_out,), dtype)))
    return params


def mlp(params, z):
    """z: (d_in,) -> scalar."""
    h = z
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[0]


def n_params(tree):
    """Total number of trainable scalars in a parameter tree."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(tree)))
