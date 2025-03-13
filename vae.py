import jax
import flax.linen as nn
import jax.numpy as jnp
from model_utils import conv3x3, conv1x1, groupnorm, sqaVAEAttention


class ResnetBlock(nn.Module):

    in_channels: int
    out_channels: int = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # super().__init__()
        in_channels = self.in_channels
        out_channels = self.out_channels
        dtype = self.dtype

        # begin of work
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = groupnorm(dtype=dtype)
        self.conv1 = conv3x3(out_channels, dtype=dtype)
        self.norm2 = groupnorm(dtype=dtype)
        self.conv2 = conv3x3(out_channels, dtype=dtype)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = conv1x1(out_channels, dtype=dtype)
        else:
            self.nin_shortcut = None
        # self.swish = nn.silu(inplace=True)

    def forward(self, x):
        print(f"In ResnetBlock, pos 0")
        hidden = x
        hidden = self.norm1(hidden)
        print(f"In ResnetBlock, pos 1")
        hidden = nn.silu(hidden)
        hidden = self.conv1(hidden)
        print(f"In ResnetBlock, pos 2")
        hidden = self.norm2(hidden)
        print(f"In ResnetBlock, pos 3")
        hidden = nn.silu(hidden)
        hidden = self.conv2(hidden)
        print(f"In ResnetBlock, pos 4")
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        print(f"In ResnetBlock, pos 5")
        return x + hidden



class Downsample(nn.Module):

    in_channels: int
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        """
        TODO: check the padding
        """
        # pad = ((0, 0), (0, 1), (0, 1), (0, 0)) # for each dimension, the pad of before and after
        # x = jnp.pad(x, pad, mode="constant", constant_values=0)
        x = nn.Conv(
            self.in_channels, 
            (3, 3), 
            (2, 2), 
            padding=((0, 1), (0, 1)), 
            dtype=self.dtype, 
            name="conv"
        )(x)
        return x


class Upsample(nn.Module):

    in_channels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, 2*H, 2*W, C), method="nearest")
        x = conv3x3(self.in_channels, dtype=self.dtype, name="conv")(x)
        return x

class VAEEncoder(nn.Module):

    ch: int = 128
    ch_mult: tuple[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    in_channels: int = 3
    z_channels: int = 16
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        ch = self.ch
        ch_mult = self.ch_mult
        num_res_blocks = self.num_res_blocks
        in_channels = self.in_channels
        z_channels = self.z_channels
        dtype = self.dtype

        # preparations
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        # self.conv_in = conv3x3(ch, dtype=dtype)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult

        hs = [conv3x3(ch, dtype=dtype, name="conv3x3_in")(x)]
        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                h = ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dtype=dtype,
                    name=f"down_{i_level}_ResBlock_{i_block}"
                )(hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = Downsample(
                    block_in, 
                    dtype=dtype, 
                    name=f"down_{i_level}_downsample"
                )(hs[-1])
                hs.append(h)
        # middle
        h = hs[-1]
        h = ResnetBlock(
            in_channels=block_in, 
            out_channels=block_in, 
            dtype=dtype, 
            name="mid_ResBlock_1"
        )(h)
        h = sqaVAEAttention(
            block_in, 
            dtype=dtype, 
            name="mid_attn_1"
        )(h)
        h = ResnetBlock(
            in_channels=block_in, 
            out_channels=block_in, 
            dtype=dtype, 
            name="mid_ResBlock_2"
        )(h)
        # end
        h = groupnorm(block_in, dtype=dtype, name="norm_out")(h)
        h = nn.silu(h)
        h = conv3x3(2*z_channels, dtype=dtype, name="conv_out")(h)
        return h