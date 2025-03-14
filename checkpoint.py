from safetensors import safe_open

# code for transfering VAE params from pytorch to jax
with safe_open("/kmh-nfs-ssd-eu-mount/data/SD3.5_pretrained_models/sd3.5_medium.safetensors", framework="pt") as f:
    keys = f.keys()
    # print(keys)
    # tensor = f.get_tensor("text_model.final_layer_norm.bias")
    # print(tensor.shape)
