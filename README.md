# jax_SD

try jax implementation of Stable Diffusion API

# official documentation (tang)

🤗 Diffusers supports Flax for super fast inference on Google TPUs, such as those available in Colab, Kaggle or Google Cloud Platform. This guide shows you how to run inference with Stable Diffusion using JAX/Flax.

Before you begin, make sure you have the necessary libraries installed:

```bash
pip install -q jax==0.3.25 jaxlib==0.3.25 flax transformers ftfy
pip install -q diffusers
```

Great, now you can import the rest of the dependencies you’ll need:

```python
import jax.numpy as jnp
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline
```

## Load a model

Flax is a functional framework, so models are stateless and parameters are stored outside of them. Loading a pretrained Flax pipeline returns both the pipeline and the model weights (or parameters). In this guide, you’ll use bfloat16, a more efficient half-float type that is supported by TPUs (you can also use float32 for full precision if you want).

```python
dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    variant="bf16",
    dtype=dtype,
)
```

## Inference

TPUs usually have 8 devices working in parallel, so let’s use the same prompt for each device. This means you can perform inference on 8 devices at once, with each device generating one image. As a result, you’ll get 8 images in the same amount of time it takes for one chip to generate a single image!

After replicating the prompt, get the tokenized text ids by calling the prepare_inputs function on the pipeline. The length of the tokenized text is set to 77 tokens as required by the configuration of the underlying CLIP text model.

```python
prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
prompt = [prompt] * jax.device_count()
prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids.shape
# (8, 77)
```

Model parameters and inputs have to be replicated across the 8 parallel devices. The parameters dictionary is replicated with `flax.jax_utils.replicate` which traverses the dictionary and changes the shape of the weights so they are repeated 8 times. Arrays are replicated using shard.

```python
# parameters
p_params = replicate(params)

# arrays
prompt_ids = shard(prompt_ids)
prompt_ids.shape
# (8, 1, 77)
```

This shape means each one of the 8 devices receives as an input a jnp array with shape (1, 77), where 1 is the batch size per device. On TPUs with sufficient memory, you could have a batch size larger than 1 if you want to generate multiple images (per chip) at once.

Next, create a random number generator to pass to the generation function. This is standard procedure in Flax, which is very serious and opinionated about random numbers. All functions that deal with random numbers are expected to receive a generator to ensure reproducibility, even when you’re training across multiple distributed devices.

The helper function below uses a seed to initialize a random number generator. As long as you use the same seed, you’ll get the exact same results. Feel free to use different seeds when exploring results later in the guide.

```python
def create_key(seed=0):
    return jax.random.PRNGKey(seed)
```

The helper function, or rng, is split 8 times so each device receives a different generator and generates a different image.

```python
rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())
```

To take advantage of JAX’s optimized speed on a TPU, pass `jit=True` to the pipeline to compile the JAX code into an efficient representation and to ensure the model runs in parallel across the 8 devices.

The first inference run takes more time because it needs to compile the code, but subsequent calls (even with different inputs) are much faster. For example, it took more than a minute to compile on a TPU v2-8, but then it takes about 7s on a future inference run!

```python
%%time
images = pipeline(prompt_ids, p_params, rng, jit=True)[0]

# CPU times: user 56.2 s, sys: 42.5 s, total: 1min 38s
# Wall time: 1min 29s
```

The returned array has shape `(8, 1, 512, 512, 3)` which should be reshaped to remove the second dimension and get 8 images of 512 × 512 × 3. Then you can use the `numpy_to_pil()` function to convert the arrays into images.

```python
from diffusers.utils import make_image_grid

images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
make_image_grid(images, rows=2, cols=4)
```

## Using different prompts

You don’t necessarily have to use the same prompt on all devices. For example, to generate 8 different prompts:

```python
prompts = [
    "Labrador in the style of Hokusai",
    "Painting of a squirrel skating in New York",
    "HAL-9000 in the style of Van Gogh",
    "Times Square under water, with fish and a dolphin swimming around",
    "Ancient Roman fresco showing a man working on his laptop",
    "Close-up photograph of young black woman against urban background, high quality, bokeh",
    "Armchair in the shape of an avocado",
    "Clown astronaut in space, with Earth in the background",
]

prompt_ids = pipeline.prepare_inputs(prompts)
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, p_params, rng, jit=True).images
images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)

make_image_grid(images, 2, 4)
```

## How does parallelization work?

The Flax pipeline in 🤗 Diffusers automatically compiles the model and runs it in parallel on all available devices. Let’s take a closer look at how that process works.

JAX parallelization can be done in multiple ways. The easiest one revolves around using the `jax.pmap` function to achieve single-program multiple-data (SPMD) parallelization. It means running several copies of the same code, each on different data inputs. More sophisticated approaches are possible, and you can go over to the JAX documentation to explore this topic in more detail if you are interested!

`jax.pmap` does two things:

1. Compiles (or `jits`) the code which is similar to `jax.jit()`. This does not happen when you call `pmap`, and only the first time the pmapped function is called.
2. Ensures the compiled code runs in parallel on all available devices.

To demonstrate, call `pmap` on the pipeline’s _generate method (this is a private method that generates images and may be renamed or removed in future releases of 🤗 Diffusers):

```python
p_generate = pmap(pipeline._generate)
```

After calling `pmap`, the prepared function `p_generate` will:
1. Make a copy of the underlying function, pipeline._generate, on each device.
2. Send each device a different portion of the input arguments (this is why it’s necessary to call the shard function). In this case, prompt_ids has shape (8, 1, 77, 768) so the array is split into 8 and each copy of `_generate` receives an input with shape (1, 77, 768).

The most important thing to pay attention to here is the batch size (1 in this example), and the input dimensions that make sense for your code. You don’t have to change anything else to make the code work in parallel.

The first time you call the pipeline takes more time, but the calls afterward are much faster. The `block_until_ready` function is used to correctly measure inference time because JAX uses asynchronous dispatch and returns control to the Python loop as soon as it can. You don’t need to use that in your code; blocking occurs automatically when you want to use the result of a computation that has not yet been materialized.

```python
%%time
images = p_generate(prompt_ids, p_params, rng)
images = images.block_until_ready()

# CPU times: user 1min 15s, sys: 18.2 s, total: 1min 34s
# Wall time: 1min 15s
```