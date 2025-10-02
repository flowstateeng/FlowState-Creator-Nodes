# 🌊🎬 FlowState WAN Studio

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

**Outputs**:
| Name | Type | Description |
| --- | --- | --- |
| `image` | `IMAGE` | The generated video frames as an image batch. |
| `latent` | `LATENT` | The final latent batch from the sampling process. |

<br/>

Video tutorials can be found on YouTube at [🌊 FlowState Creator Suite Playlist](https://www.youtube.com/playlist?list=PLopF-DMGUFkTulZRkSpRmKFcTENKFicws) *(Coming Soon)*
