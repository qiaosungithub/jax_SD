import torch, jax
import jax.numpy as jnp
import einops
import flax.linen as nn
from jax import random

class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        # self.norm = Normalize(in_channels, dtype=dtype, device=device)
        # self.q = torch.nn.Conv2d(
        #     in_channels,
        #     in_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     dtype=dtype,
        #     device=device,
        # )
        # self.k = torch.nn.Conv2d(
        #     in_channels,
        #     in_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     dtype=dtype,
        #     device=device,
        # )
        # self.v = torch.nn.Conv2d(
        #     in_channels,
        #     in_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     dtype=dtype,
        #     device=device,
        # )
        # self.proj_out = torch.nn.Conv2d(
        #     in_channels,
        #     in_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     dtype=dtype,
        #     device=device,
        # )

    def forward(self, x):
        x = torch.arange(12).reshape(1, 3, 2, 2).float()
        # hidden = self.norm(x)
        # q = self.q(hidden)
        # k = self.k(hidden)
        # v = self.v(hidden)
        q = torch.arange(12).reshape(1, 3, 2, 2).float()
        k = torch.arange(12).reshape(1, 3, 2, 2).float()
        v = torch.arange(12).reshape(1, 3, 2, 2).float()
        q = q + 1
        k = k - 4
        v = v - 2
        b, c, h, w = q.shape
        q, k, v = map(
            lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(),
            (q, k, v),
        )
        print(f"q: {q}")
        hidden = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default
        # another implementation
        attn_weight = q @ (k.transpose(-2, -1)) * (1 / (c ** 0.5))
        attn_weight = torch.softmax(attn_weight, dim=-1)
        value = attn_weight @ v
        print(f"value: {value}")
        # print(f"hidden: {hidden}")
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        # hidden = self.proj_out(hidden)
        return hidden + x
    
A = AttnBlock(3)
print(A.forward(0))


class AttnBlock(nn.Module):
    """
    Input shape: (B, H, W, C) , instead of (B, N, C)
    Output shape: (B, H, W, C)
    TODO: check whether this is correct. I believe it's correct. This version is a combination of our DiT attention and SD3 implementation.
    """

    dim: int
    num_heads: int = 1
    qkv_bias: bool = True
    # norm_layer: nn.Module = bomb
    # linear_layer: nn.Module = bomb
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.num_heads == 1, "only support 1 head"
        num_heads = self.num_heads; dim = self.dim; qkv_bias = self.qkv_bias
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # # self.qkv = self.linear_layer(dim, dim * 3, bias=qkv_bias)
        # self.q = conv1x1(dim, dtype=self.dtype)
        # self.k = conv1x1(dim, dtype=self.dtype)
        # self.v = conv1x1(dim, dtype=self.dtype)

        # self.norm = groupnorm(dtype=self.dtype)

        # self.proj_out = conv1x1(dim, dtype=self.dtype)

    def flat_for_attention(self, x: jnp.ndarray):
        raise NotImplementedError
        B, H, W, C = x.shape
        N = H * W
        NH = self.num_heads
        HD = self.head_dim
        x = x.reshape(B, N, C)
        return x

    def __call__(self, x):
        skip = x
        x = jnp.arange(12).reshape(1, 3, 2, 2).astype(jnp.float32).transpose(0, 2, 3, 1)
        skip = x
        B, H, W, C = x.shape
        # x = self.norm(x)
        q = x
        k = x
        v = x
        q = q + 1
        k = k - 4
        v = v - 2
        # q = self.q(x)
        # k = self.k(x)
        # v = self.v(x) # shape (B, H, W, C)
        N = H * W
        NH = self.num_heads
        HD = self.dim # =dim for 1 head case
        # transpose to 1, B, NH, N, HD
        q = q.reshape(1, B, N, NH, HD).transpose(0, 1, 3, 2, 4)
        k = k.reshape(1, B, N, NH, HD).transpose(0, 1, 3, 2, 4)
        v = v.reshape(1, B, N, NH, HD).transpose(0, 1, 3, 2, 4)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4) # 3, B, num_heads, N, head_dim
        # q, k, v = qkv.unbind(0)
        # q, k, v = jnp.split(qkv, 3, axis=0) # k.shape: (1, 2, 6, 64, 64)
        # q, k = self.q_norm(q), self.k_norm(k)
        print(f"q: {q}")

        q = q * (self.dim ** -0.5)
        attn = q @ (k.transpose(0, 1, 2, 4, 3)) # 1, B, num_heads, N, N
        attn = jax.nn.softmax(attn, axis=-1) # 1, B, num_heads, N, N
        # attn = self.attn_drop(attn)
        x = attn @ v # 1, B, num_heads, N, head_dim
        
        x = x[0].transpose(0, 2, 1, 3).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        x = x.reshape(B, H, W, C)
        print(f"final result: {x}")
        # x = self.proj_out(x)
        return x + skip

def initialized(key, shape, model):
    """
    Initialize the model, and return the model parameters.
    """
    # input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)
    
    key, 东西 = random.split(key)

    print("Initializing params...")
    variables = init({"params": key}, jnp.ones(shape, model.dtype))
    if "batch_stats" not in variables:
        variables["batch_stats"] = {}
    print("Initializing params done.")
    return None

A = AttnBlock(3)
rng = random.key(0)
_ = initialized(rng, (1, 2, 2, 3), A)
# init
print(A(0))
