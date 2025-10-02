# 🌊👩🏻‍🍳 FlowState Chef

The powerhouse of the Chef pipeline. This node executes a two-stage, instruction-based image editing and refinement workflow.

- An all-in-one image editing pipeline that seamlessly combines:
- A Qwen model for executing direct, instruction-based edits on input images.
- A FLUX model for high-quality refinement of the initial edit.
- Intelligent stage caching to avoid re-running stages where parameters haven't changed, saving time.
- Full control over both stages, including models, LoRAs, VAEs, samplers, and more.

<p align="center">
<img width='650' src='https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/imgs/FlowState%20Chef.png' alt='FS Chef Image'/>
</p>

**Inputs**
| Name | Type | Description |
| --- | --- | --- |
| `chef_ingredients` | `ANY` | The output from the FlowState Chef Ingredients node. |
| `qwen_model` | `STRING` | **Qwen Model**: The Qwen UNET model used for the initial editing stage. |
| `flux_model_filetype` | `STRING` | **FLUX File Type**: Specify whether the FLUX model is a full checkpoint or a UNET-only file. |
| `flux_model` | `STRING` | **FLUX Model**: The FLUX model used for the final refinement stage. |
| `weight_dtype` | `STRING` | **Weight Datatype (DType)**: The data type for model weights (e.g., float16). |
| `qwen_optimization_lora` | `STRING` | **Qwen Opt. LoRA**: An optimization LoRA applied only to the Qwen model. |
| `qwen_style_lora` | `STRING` | **Qwen Style LoRA**: A style LoRA applied only to the Qwen model. |
| `qwen_style_lora_strength`| `FLOAT` | **Qwen LoRA Strength**: The strength of the Qwen style LoRA. |
| `flux_style_lora` | `STRING` | **FLUX Style LoRA**: A style LoRA applied only to the FLUX model. |
| `flux_style_lora_strength`| `FLOAT` | **FLUX LoRA Strength**: The strength of the FLUX style LoRA. |
| `sage_attention` | `STRING` | **Sage Attention Mode**: The type of Sage Attention patching to apply. |
| `qwen_clip` | `STRING` | **Qwen CLIP**: The CLIP model compatible with the selected Qwen model. |
| `qwen_vae` | `STRING` | **Qwen VAE**: The VAE used for the Qwen stage. |
| `flux_clip_1` | `STRING` | **FLUX CLIP 1**: The first CLIP model for the FLUX stage. |
| `flux_clip_2` | `STRING` | **FLUX CLIP 2**: The second CLIP model for the FLUX stage. |
| `flux_vae` | `STRING` | **FLUX VAE**: The VAE used for the FLUX stage. |
| `tiled_decode` | `BOOLEAN`| Ti**led Decode**: Enables tiled VAE decoding for higher resolutions to conserve VRAM. |
| `resolution` | `STRING` | **Resolution Selector**: Select a preset resolution or 'Custom'. |
| `orientation` | `STRING` | **Orientation Selector**: Swaps the aspect ratio of the selected preset. |
| `latent_type` | `STRING` | **Latent Type**: The source for the initial latent (e.g., Empty, from Image). |
| `custom_width` | `INT` | **Custom Width**: Defines the width of the output when resolution is 'Custom'. |
| `custom_height` | `INT` | **Custom Height**: Defines the height of the output when resolution is 'Custom'. |
| `custom_batch_size` | `INT` | **Batch Size**: The number of images to generate. |
| `seed` | `INT` | **Seed**: The seed used to generate the initial random noise. |
| `sampling_algorithm`| `STRING` | **Sampling Algorithm**: The sampler to use for the diffusion process. |
| `scheduling_algorithm`| `STRING` | **Scheduling Algorithm**: The scheduler to use for the diffusion process. |
| `qwen_steps` | `INT` | **Qwen Steps**: The number of steps for the Qwen sampling stage. |
| `qwen_denoise` | `FLOAT` | **Qwen Denoise**: The denoise amount for the Qwen sampling stage. |
| `flux_guidance` | `FLOAT` | **FLUX Guidance**: The guidance scale (CFG) for the FLUX stage. |
| `flux_steps` | `INT` | **FLUX Steps**: The number of steps for the FLUX refinement stage. |
| `flux_denoise` | `FLOAT` | **FLUX Denoise**: The denoise amount for the FLUX refinement stage. |

**Outputs**
| Name | Type | Description |
| --- | --- | --- |
| `image` | `IMAGE` | The final generated image batch after FLUX refinement. |
| `latent` | `LATENT` | The final latent batch from the FLUX refinement process. |

<br/>

Video tutorials can be found on YouTube at [🌊 FlowState Creator Suite Playlist](https://www.youtube.com/playlist?list=PLopF-DMGUFkTulZRkSpRmKFcTENKFicws) *(Coming Soon)*
