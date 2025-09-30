### [🌊🚒 FlowState Flux Engine](https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/FlowState_FluxEngine.py)
Streamline your entire generation workflow with the FlowState Flux Engine.

An all-in-one node that seamlessly combines:
- Model, VAE, and CLIP loading (with Model Persistence to prevent subsequent reloads.)
- Prompt conditioning & guidance.
- Sage Attention model patching (for users with that capability)
- The integrated FlowState Latent Source for flexible resolution selection & easy i2i style transfer.

Reduce your node clutter and go from concept to final image faster than ever within a single, powerful engine.
<p align="center">
  <img width='650' src='https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/imgs/FlowState%20Flux%20Engine.png' alt='FS Latent Source Image'/>
</p>

**Inputs:**
| Name | Type | Description |
| --- | --- | --- |
| `model_filetype` | `STRING` | **Model File Type**: The type of model file to load. |
| `model_name` | `STRING` | **Full Diffusion Model List**: List of all available Diffusion Models. |
| `weight_dtype` | `STRING` | **Weight Datatype (DType)**: The data type to be used for your models weights. |
| `sage_attention`| `STRING` | **Sage Attention Mode**: The type of Sage Attention to use. |
| `lora_model`| `STRING` | **LoRA Model**: Select a LoRA model to use. |
| `lora_strength`| `STRING` | **LoRA Strength**: If using a LoRA, specify strength. |
| `clip_1_name` | `STRING` | **CLIP / Text Encoder List**: List of available Text Encoders and CLIP models. |
| `clip_2_name` | `STRING` | **CLIP / Text Encoder List**: List of available Text Encoders and CLIP models. |
| `vae_name` | `STRING` | **VAE List**: List of available Variational Autoencoders (VAE). |
| `resolution` | `STRING` | **Resolution Selector**: Select custom to use the entered width & height, or select a resolution. |
| `orientation` | `STRING` | **Orientaion Selector**: Resolutions given in horizontal orientation. Select vertical to swap resolution aspect ratio. |
| `latent_type` | `STRING` | **Latent Type**: Your choice of an empty latent (all zeros) or an image as a latent. |
| `custom_width` | `INT` | **Width**: Defines the width of the image. |
| `custom_height` | `INT` | **Height**: Defines the height of the image. |
| `custom_batch_size`| `INT` | **Custom Batch Size**: The number of images you want to generate. |
| `image` | `STRING` | **Uploaded Image**: Path to the image file to be used when `latent_type` is 'Uploaded Image'. |
| `seed` | `INT` | **Seed**: Seed used to generate inital random noise. |
| `sampling_algorithm`| `STRING` | **Sampling Algorithm**: List of available Sampling Algorithms. |
| `scheduling_algorithm`|`STRING` | **Scheduling Algorithm**: List of available Scheduling Algorithms. |
| `guidance` | `FLOAT` | **Guidance**: Defines the guidance scale. |
| `steps` | `INT` | **Steps**: Defines the number of steps to take in the sampling process. |
| `denoise` | `FLOAT` | **Sampler Denoise Amount**: The amount of denoising applied. |
| `prompt` | `STRING` | **Positive Prompt**: Positive text prompt describing your desired output. |
| `input_img` (optional)| `IMAGE` | **Input Image**: An optional image passed from another node. |

**Outputs:**
| Name | Type | Description |
| --- | --- | --- |
| `model` | `MODEL` | The selected Diffusion Model. |
| `clip` | `CLIP` | The selected CLIP. |
| `vae` | `VAE` | The selected VAE. |
| `image` | `IMAGE` | The image batch. |
| `latent` | `LATENT` | The latent batch. |

**Limitations:**
* Only tested on core Flux models: Dev, Schnell, Krea, Fill, Chroma, Kontext. Cannot guarantee compatiblity with versions from other developers such as De-Distilled, etc.

<br/>

Video tutorials can be found on YouTube at [🌊 FlowState Creator Suite Playlist](https://www.youtube.com/playlist?list=PLopF-DMGUFkTulZRkSpRmKFcTENKFicws) *(Coming Soon)*
