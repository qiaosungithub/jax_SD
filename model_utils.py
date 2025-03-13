import flax.linen as nn
import jax.numpy as jnp
import jax
import jax.random as jr
from math import sqrt
from functools import partial

bomb = lambda : exec('raise ValueError')

def conv1x1(out_channels, dtype=jnp.float32, name=None):
    return nn.Conv(out_channels, (1, 1), (1, 1), 'SAME', dtype=dtype, name=name)

def conv3x3(out_channels, dtype=jnp.float32, name=None):
    return nn.Conv(out_channels, (3, 3), (1, 1), 'SAME', dtype=dtype, name=name)

def groupnorm(num_groups=32, dtype=jnp.float32, name=None):
    return nn.GroupNorm(
        num_groups=num_groups,
        # num_channels=in_channels,
        epsilon=1e-6,
        use_bias=True,
        dtype=dtype,
        name=name
    )

class AttnBlock(nn.Module):
    """
    Input shape: (B, H, W, C) , instead of (B, N, C)
    Output shape: (B, H, W, C)
    TODO: check whether this is correct. I believe it's correct. This version is a combination of our DiT attention and SD3 implementation.
    """

    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    # norm_layer: nn.Module = bomb
    linear_layer: nn.Module = bomb
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        num_heads = self.num_heads; dim = self.dim; qkv_bias = self.qkv_bias
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = self.linear_layer(dim, dim * 3, bias=qkv_bias)

        self.norm = groupnorm(dtype=self.dtype)

        self.proj = self.linear_layer(dim, dim, bias=True)

    def forward(self, x):
        skip = x
        B, H, W, C = x.shape
        x = self.norm(x)
        x = x.reshape(B, H * W, C)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4) # 3, B, num_heads, N, head_dim
        # q, k, v = qkv.unbind(0)
        q, k, v = jnp.split(qkv, 3, axis=0) # k.shape: (1, 2, 6, 64, 64)
        # q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(0, 1, 2, 4, 3) # 1, B, num_heads, i, j
        attn = jax.nn.softmax(attn, axis=-1)
        # attn = self.attn_drop(attn)
        x = attn @ v # (1, 2, 6, 64, 64) ; (1, B, num_heads, i, j) @ (1, B, num_heads, j, head_dim)
        
        x = x[0].transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        x = x.reshape(B, H, W, C)
        return x + skip

class TorchLinear(nn.Module):
    """Written by ZHH. This is correct."""

    in_features: int
    out_features: int
    bias: bool = True
    weight_init: str = 'torch' # options: 'torch', 'xavier_uniform', '0.02', 'zeros
    bias_init: str = 'torch' # options: 'torch', 'zeros'

    def setup(self):
        if self.weight_init == 'torch':
            weight_initializer = nn.initializers.variance_scaling(scale=1/3.0, mode='fan_in', distribution='uniform')
        elif self.weight_init == 'xavier_uniform':
            weight_initializer = nn.initializers.xavier_uniform()
        elif self.weight_init == '0.02':
            weight_initializer = lambda key, shape, dtype: jr.normal(key, shape) * 0.02
        elif self.weight_init == 'zeros':
            weight_initializer = nn.initializers.zeros
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")
        
        if self.bias_init == 'torch':
            bias_initializer = lambda key, shape, dtype: jr.uniform(key, shape, minval=-sqrt(1/self.in_features), maxval=sqrt(1/self.in_features))
        elif self.bias_init == 'zeros':  
            bias_initializer = nn.initializers.zeros
        else:
            raise ValueError(f"Invalid bias_init: {self.bias_init}")

        self._flax_linear = nn.Dense(features=self.out_features, use_bias=self.bias, kernel_init=weight_initializer, bias_init=bias_initializer)

    def __call__(self, x):
        return self._flax_linear(x)

DiTLinear = partial(TorchLinear, weight_init='xavier_uniform', bias_init='zeros')
sqaVAEAttention = partial(AttnBlock, linear_layer=DiTLinear, num_heads=1)