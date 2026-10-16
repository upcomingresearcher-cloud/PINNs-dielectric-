"""Fully connected tanh networks (protocol Sec. 5): Glorot-normal weights, zero biases."""
import jax, jax.numpy as jnp


def init_mlp(key, d_in, width, depth, dtype):
    sizes = [d_in] + [width] * depth + [1]
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        params.append((jax.random.normal(k, (fan_in, fan_out), dtype) * std.astype(dtype),
                       jnp.zeros((fan_out,), dtype)))
    return params


def mlp(params, z):
    """z: (d_in,) -> scalar."""
    h = z
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[0]


def n_params(tree):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(tree)))
