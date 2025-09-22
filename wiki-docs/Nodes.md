## 🖧 Current Nodes

### [🌊🎬 FlowState WAN Studio](https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/FlowState_WANStudio.py)
An all-in-one studio for generating high-quality video using a dual-model WAN pipeline.

A complete video generation pipeline that seamlessly combines:
- Dual-model loading for a two-stage, refiner-style sampling workflow.
- Advanced LoRA patching with separate controls for high-noise, low-noise, and style LoRAs.
- Integrated Sage Attention model patching.
- Flexible video sizing with presets, custom resolution, and img2vid from a starting frame.
- Simplify your complex video workflows and orchestrate the entire generation process within a single, powerful studio.

<p align="center">
<img width='650' src='https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/imgs/FlowState%20WAN%20Studio.png' alt='FS WAN Studio Image'/>
</p>

**Inputs:**
| Name | Type | Description |
| --- | --- | --- |
| `high_noise_model_name` | `STRING` | **High-Noise Model**: The diffusion model used for the first stage of sampling. |
| `low_noise_model_name` | `STRING` | **Low-Noise Model**: The diffusion model used for the second/refiner stage of sampling. |
| `weight_dtype` | `STRING` | **Weight Datatype (DType)**: The data type for your model weights (e.g., float16). |
| `sage_attention`| `STRING` | **Sage Attention Mode**: The type of Sage Attention patching to apply to the models. |
| `high_noise_lora`| `STRING` | **High-Noise LoRA**: A LoRA applied only to the high-noise model. |
| `low_noise_lora`| `STRING` | **Low-Noise LoRA**: A LoRA applied only to the low-noise model. |
| `style_lora`| `STRING` | **Style LoRA**: A LoRA applied to both models. |
| `clip_name` | `STRING` | **CLIP / Text Encoder**: List of available Text Encoders and CLIP models. |
| `vae_name` | `STRING` | **VAE List**: List of available Variational Autoencoders (VAE). |
| `resolution` | `STRING` | **Resolution Selector**: Select a preset resolution or 'Custom' to use manual width/height. |
| `orientation` | `STRING` | **Orientation Selector**: Swaps the aspect ratio of the selected preset resolution. |
| `custom_width` | `INT` | **Custom Width**: Defines the width of the video when resolution is 'Custom'. |
| `custom_height` | `INT` | **Custom Height**: Defines the height of the video when resolution is 'Custom'. |
| `num_video_frames` | `INT` | **Number of Video Frames**: The total number of frames in the generated video. |
| `seed` | `INT` | **Seed**: The seed used to generate the initial random noise. |
| `sampling_algorithm`| `STRING` | **Sampling Algorithm**: The sampler to use for the diffusion process. |
| `scheduling_algorithm`|`STRING` | **Scheduling Algorithm**: The scheduler to use for the diffusion process. |
| `steps` | `INT` | **Steps**: The total number of steps for the sampling process. |
| `pos_prompt` | `STRING` | **Positive Prompt**: The positive text prompt describing the desired output. |
| `neg_prompt` | `STRING` | **Negative Prompt**: The negative text prompt for things to avoid. |
| `starting_frame` (optional)| `IMAGE` | **Starting Frame**: An optional image to use as the first frame for img2vid. |
| `clip_vision` (optional)| `CLIP_VISION` | **CLIP Vision Output**: An optional CLIP Vision output to guide the generation. |

Outputs:
| Name | Type | Description |
| --- | --- | --- |
| `image` | `IMAGE` | The generated video frames as an image batch. |
| `latent` | `LATENT` | The final latent batch from the sampling process. |

---

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

---

### [🌊🌱 FlowState Latent Source](https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/FlowState_LatentSource.py)
Simplify your latent options and clean up your workflow with FlowState Latent Source.

A simple switch to select between:
 - an empty latent
 - input image as a latent
 - uploaded image as a latent
 
Vary the denoise on your sampler to control the amount of style transfer you want from your input image in your generated images.
<p align="center">
  <img width='650' src='https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/imgs/FlowState%20Latent%20Source.png' alt='FS Latent Source Image'/>
</p>

**Inputs:**
| Name | Type | Description |
| --- | --- | --- |
| `resolution` | `STRING` | **Resolution Selector**: Select custom to use the entered width & height, or select a resolution. |
| `orientation` | `STRING` | **Orientaion Selector**: Resolutions given in horizontal orientation. Select vertical to swap resolution aspect ratio. |
| `latent_type` | `STRING` | **Latent Type**: Your choice of an empty latent (all zeros) or an image as a latent. |
| `custom_width` | `INT` | **Width**: Defines the width of the image. |
| `custom_height` | `INT` | **Height**: Defines the height of the image. |
| `custom_batch_size` | `INT` | **Custom Batch Size**: The number of images you want to generate. |
| `image` | `STRING` | **Uploaded Image**: Path to the image file to be used when `latent_type` is 'Uploaded Image'. |
| `vae` | `VAE` | **Variational AutoEncoder (VAE)**: The VAE model used for encoding and decoding images. |
| `input_img` (optional) | `IMAGE` | **Input Image**: An optional image passed from another node, used when `latent_type` is 'Input Image'. |

**Outputs:**
| Name | Type | Description |
| --- | --- | --- |
| `Latent Image` | `LATENT` | The latent image batch. |

<br/>

Video tutorials can be found on YouTube at [🌊 FlowState Creator Suite Playlist](https://www.youtube.com/playlist?list=PLopF-DMGUFkTulZRkSpRmKFcTENKFicws) *(Coming Soon)*
