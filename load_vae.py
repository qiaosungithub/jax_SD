import flax.linen as nn
import jax.numpy as jnp
import jax, wandb, os, orbax
from jax import random
from flax.training import orbax_utils

from models.vae import VAEEncoder, VAEDecoder

from utils.logging_utils import log_for_0

class SDVAE(nn.Module):

    dtype: jnp.dtype=jnp.float32
    def setup(self):
        self.encoder = VAEEncoder(dtype=self.dtype)
        self.decoder = VAEDecoder(dtype=self.dtype)

    def decode(self, latent): # we need to cast to bf16
        return self.decoder(latent)

    def encode(self, image, rng): # we need to cast to bf16
        hidden = self.encoder(image)
        mean, logvar = jnp.split(hidden, 2, axis=-1)
        logvar = jnp.clip(logvar, -30.0, 20.0)
        std = jnp.exp(0.5 * logvar)
        noise = random.normal(rng, mean.shape, dtype=mean.dtype)
        return mean + std * noise

    def __call__(self, image, rng):
        # for init model
        latent = self.encode(image, rng)
        return self.decode(latent)

def initialized(key, shape, model):
    """
    Initialize the model, and return the model parameters.
    """
    # input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)
    
    key, 东西 = random.split(key)

    log_for_0("Initializing params...")
    variables = init({"params": key}, jnp.ones(shape, model.dtype), 东西)
    if "batch_stats" not in variables:
        variables["batch_stats"] = {}
    log_for_0("Initializing params done.")
    return variables["params"], variables["batch_stats"]

if __name__ == "__main__":

    ################# init #################
    if jax.process_index() == 0:
        workdir = os.getcwd()
        wandb.init(project="SD3", dir=workdir, tags=["vae", "debug"])
        # wandb.config.update(config.to_dict())
        # ka = re.search(r"kmh-tpuvm-v[234]-(\d+)(-preemptible)?-(\d+)", workdir).group()
        # wandb.config.update({"ka": ka})
    rank = jax.process_index()
    rng = random.key(0)

    model = SDVAE(dtype=jnp.bfloat16)
    # load model from .safetensors file

    # wan
    params, batch_stats = initialized(rng, (1, 128, 128, 3), model)