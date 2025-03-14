import flax.linen as nn
import jax.numpy as jnp
import jax, wandb, os
from jax import random
from safetensors import safe_open
import matplotlib.pyplot as plt
from PIL import Image
# from flax.training import orbax_utils

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

    # init VAE model
    log_for_0("Initializing model...")
    params, batch_stats = initialized(rng, (1, 128, 128, 3), model)
    log_for_0("Initializing model done.")

    # # try whether it can run with different sizes
    # x = model.apply({"params": params, "batch_stats": batch_stats}, jnp.ones((1, 256, 256, 3), model.dtype), rng, method=model.encode)
    # print(x.shape)

    # def print_tree(tree, indent=""):
    #     for k, v in tree.items():
    #         if isinstance(v, dict):
    #             print(f"{indent}{k}:")
    #             print_tree(v, indent + "  ")
    #         else:
    #             print(f"{indent}{k}: {v.shape}")

    # print_tree(params)
    # print_tree(batch_stats)

    assert batch_stats == {}, "batch_stats should be empty."
    jax_checkpoint_path = "/kmh-nfs-ssd-eu-mount/data/SD3.5_pretrained_models/sd3.5_medium_jax.safetensors"

    # with safe_open(jax_checkpoint_path, framework="flax") as f:
    #     # for each leaf of params, read from the file
    #     def fill_params(p, path=""): # p is a pytree
    #         for k, v in p.items():
    #             if isinstance(v, dict):
    #                 fill_params(v, path + k + ".")
    #             else:
    #                 tensor = f.get_tensor(path + k)
    #                 assert tensor is not None, f"tensor is None, path: {path + k}"
    #                 p[k] = tensor

    #     fill_params(params)
    #     print("Load params done.")

    # run an encode + decode step for sanity check
    image_path = "/kmh-nfs-ssd-eu-mount/code/qiao/work/jax_SD/test_image/0/0/4.png"
    # load the image with PIL
    image = plt.imread(image_path)
    image = image[:, :, :3]
    image = 2 * image - 1
    # image = Image.open(image_path).convert("RGB")
    image = jnp.array(image)
    print(image.shape)
    image = image[None, ...]
    image = image.astype(jnp.bfloat16)
    x = model.apply({"params": params, "batch_stats": batch_stats}, image, rng)
    # x = image # for debug
    # save the image with plt
    x = x[0]
    print(x.shape)
    x = x.astype(jnp.float32)
    x = (x+1)/2
    x = jnp.clip(x, 0, 1)
    x = jax.device_get(x)
    plt.imsave("test.png", x)