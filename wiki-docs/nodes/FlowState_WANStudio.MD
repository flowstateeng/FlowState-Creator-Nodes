# 🌊🍿 FlowState WAN Studio

An all-in-one studio for generating high-quality video using a dual-model WAN pipeline.

A complete video generation pipeline that seamlessly combines:
- Dual-model loading for a two-stage, refiner-style sampling workflow.
- Advanced LoRA patching with separate controls for high-noise, low-noise, and style LoRAs.
- Integrated Sage Attention model patching.
- Flexible video sizing with presets, custom resolution, and img2vid from a starting frame.
- Simplify your complex video workflows and orchestrate the entire generation process within a single, powerful studio.

<p align="center">
<img width='650' src='https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/imgs/FlowState%20WAN%20Studio.webp' alt='FS WAN Studio Image'/>
</p>

**Inputs:**
| Name | Type | Description |
| --- | --- | --- |
| `high_noise_model_name` | `STRING` | **High-Noise Model**: The diffusion model used for the first stage of sampling. |
| `low_noise_model_name` | `STRING` | **Low-Noise Model**: The diffusion model used for the second/refiner stage of sampling. |
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
| `fixed_output` | `BOOLEAN` | **Fixed Output**: If set to `true` this will fix the seed. If `false` random seeds will be used. |
| `fps` | `INT` | **Frames Per Second**: This sets the FPS of the output video. |
| `save_video` | `BOOLEAN` | **Save Video**: Whether or not to save the video. Leave this off until you have a result you like & then turn on to save. |
| `filename_prefix` | `STRING` | **Filename Prefix**: The name of the video file. The current timestamp will be appended to the filename to help with organization. |
| `starting_frame` (optional)| `IMAGE` | **Starting Frame**: An optional image to use as the first frame for img2vid. |
| `audio` (optional)| `AUDIO` | **Starting Frame**: An optional audio track to use with the video. |

**Outputs**:
| Name | Type | Description |
| --- | --- | --- |
| `video` | `VIDEO` | The generated video. |

---

## Usage Notes

* Intelligent stage execution - If you generate a video with the `save_video` turned off, for example, you can turn it on and run again to save. The video will not be generated again. It will simply save video you've already generated. The same is true of the `fps` and `filename_prefix` options. Changing these will not cause the video to generate again, as they have no bearing on the generation process.

* Intelligent model loading - The video models do not need to load on every run. WAN Studio saves the models loaded on the first run and uses them for subsequent runs. If you change a LoRA, that model will need to be repatched, and will load again. Otherwise, loaded models will persist for all runs. Models are intelligently offloaded to system RAM as needed to free up VRAM for the generation process, but they will not need to be loaded from disk other than on the first run, unless a new patch is needed.

<br/>

Video tutorials can be found on YouTube at [🌊 FlowState Creator Suite Playlist](https://www.youtube.com/playlist?list=PLopF-DMGUFkTulZRkSpRmKFcTENKFicws) *(Coming Soon)*
