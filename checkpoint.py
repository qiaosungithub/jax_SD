from safetensors import safe_open
import torch, numpy
import jax.numpy as jnp

def vae_name_map(name, tensor:torch.Tensor):
    """
    orginial name -> new name
    example: 
        decoder.conv_in.weight
        torch.Size([512, 16, 3, 3])
    ->
        decoder.conv_in.kernel
        (3, 3, 16, 512)
    """
    # deal with many cases
    path = name.split(".")
    if path[-1] == "weight":
        path[-1] = "kernel"
        assert tensor.ndim == 4, f"tensor.shape: {tensor.shape}"
        tensor = tensor.numpy().transpose(2, 3, 1, 0)
        tensor = jnp.array(tensor)
    elif path[-1] == "bias":
        assert tensor.ndim == 1, f"tensor.shape: {tensor.shape}"
        tensor = tensor.numpy()
        tensor = jnp.array(tensor)
    ########## stop here !!!!!!!!!!!

# code for transfering VAE params from pytorch to jax
with safe_open("/kmh-nfs-ssd-eu-mount/data/SD3.5_pretrained_models/sd3.5_medium.safetensors", framework="pt") as f:
    keys = f.keys()
    # print(keys)
    # tensor = f.get_tensor("text_model.final_layer_norm.bias")
    # print(tensor.shape)
    prefix="first_stage_model."
    for k in keys:
        if k.startswith(prefix):
            print(k[len(prefix):])
            print(f.get_tensor(k).shape)

        