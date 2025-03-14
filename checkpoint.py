from safetensors import safe_open
import torch, numpy
import jax.numpy as jnp
from safetensors.numpy import save_file

def vae_name_map(name:str, tensor:torch.Tensor):
    """
    orginial name -> new name
    example: 
        decoder.conv_in.weight
        torch.Size([512, 16, 3, 3])
    ->
        decoder.conv_in.kernel
        (3, 3, 16, 512)

    note that numpy does not support bf16. So first convert to torch f32, to numpy, then to jax bf16
    """
    # deal with many cases
    path = name.split(".")
    if path[-1] == "weight": # can be weight or scale
        if path[-2].startswith("norm"):
            path[-1] = "scale"
            assert tensor.ndim == 1, f"tensor.shape: {tensor.shape}"
            tensor = tensor.to(torch.float32).numpy()
            tensor = jnp.array(tensor).astype(jnp.bfloat16)
        else:
            assert path[-2].startswith("conv") or path[-2] in ["k", 'q', 'v', 'proj_out', 'nin_shortcut'], f"unexpected path[-2]: {path[-2]}"
            path[-1] = "kernel"
            assert tensor.ndim == 4, f"tensor.shape: {tensor.shape}"
            tensor = tensor.to(torch.float32).numpy().transpose(2, 3, 1, 0)
            tensor = jnp.array(tensor).astype(jnp.bfloat16)
    elif path[-1] in ["bias"]:
        assert tensor.ndim == 1, f"tensor.shape: {tensor.shape}"
        # print(tensor)
        tensor = tensor.to(torch.float32).numpy()
        tensor = jnp.array(tensor).astype(jnp.bfloat16)
    # change name
    # mid.block_1 -> mid_ResBlock_1
    if path[1] == "mid" and path[2].startswith("block_"):
        path[1] = f"mid_ResBlock_" + path[2][6:]
        path.pop(2)
    # mid.attn_1 -> mid_attn_1
    if path[1] == "mid" and path[2].startswith("attn_"):
        path[1] = f"mid_attn_" + path[2][5:]
        path.pop(2)
    # up.0.block.0 -> up_0_ResBlock_0
    if path[1] in ["up", 'down'] and path[3] == "block":
        path[1] = f"{path[1]}_{path[2]}_ResBlock_{path[4]}"
        path.pop(2)
        path.pop(2)
        path.pop(2)
    # up.1.upsample -> up_1_upsample
    if path[1] in ["up", 'down'] and path[3] in ["upsample", "downsample"]:
        path[1] = f"{path[1]}_{path[2]}_{path[3]}"
        path.pop(2)
        path.pop(2)
    return ".".join(path), tensor

checkpoint_to_save = {}

# code for transfering VAE params from pytorch to jax
with safe_open("/kmh-nfs-ssd-eu-mount/data/SD3.5_pretrained_models/sd3.5_medium.safetensors", framework="pt") as f:
    keys = f.keys()
    # print(keys)
    # tensor = f.get_tensor("text_model.final_layer_norm.bias")
    # print(tensor.shape)
    prefix="first_stage_model."
    for k in keys:
        if k.startswith(prefix):
            # print(k[len(prefix):])
            # print(f.get_tensor(k).shape)
            kk = k[len(prefix):]
            tensor = f.get_tensor(k)
            new_name, new_tensor = vae_name_map(kk, tensor)
            print(new_name)
            print(new_tensor.shape)
            checkpoint_to_save[new_name] = new_tensor

# save the checkpoint by jnp array with safetensors
path = "/kmh-nfs-ssd-eu-mount/data/SD3.5_pretrained_models/sd3.5_medium_jax.safetensors"
save_file(checkpoint_to_save, path)
print(f"Saved the checkpoint to {path}")