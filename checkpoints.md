This file is for how to use checkpoints in jax.

# Pytree

A pytree can consist of tuples, dicts, ...

Flax checkpoints are based on pytrees.

# Save checkpoint

```py
import jax.numpy as jnp
from flax.training import checkpoints

ckpt = {"model": {"kernel": jnp.ones((3, 3)), "bias": jnp.zeros(3)}, "data": {"x": jnp.ones(3)}, "step": 6, "wgt": (4, 6)}

checkpoints.save_checkpoint(ckpt_dir='/tmp/test_1',
                            target=ckpt,
                            step=0,
                            overwrite=True,
                            keep=2)
```

Any pytree can be saved as a checkpoint.

However, it seems that when load it, it will become all dicts.

# Load checkpoint

## No target loading

```py
raw = checkpoints.restore_checkpoint(ckpt_dir='/tmp/test_1', target=None)

print(raw)
```

Output:
```py
{'data': {'x': Array([1., 1., 1.], dtype=float32)}, 'model': {'bias': Array([0., 0., 0.], dtype=float32), 'kernel': Array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]], dtype=float32)}, 'step': 6, 'wgt': {'0': 4, '1': 6}}
```

## With target loading

```py
S = {"model": {"kernel": None}, "data": {"x": jnp.zeros(2)}, "step": 4}

raw = checkpoints.restore_checkpoint(ckpt_dir='/tmp/test_1', target=S)

print(raw)
```

Output:
```py
{'model': {'kernel': None}, 'data': {'x': Array([1., 1., 1.], dtype=float32)}, 'step': 6}
```

**Conclusions**:
- `None` cannot be replaced by a pytree. (ignored)
- For keys that are not in the target, they will be ignored.
- For scalars, they will also be restored.
- For arrays with different shapes, they will also be restored.
- If trying to restore an array to a scalar, it will raise an error.
- If trying to restore a scalar to an array, it will become a new 0-dim array. This is tang.