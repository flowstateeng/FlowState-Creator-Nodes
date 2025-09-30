### [🌊👌 FlowState Simple Latent](https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/FlowState_LatentSource.py)
Simplify your latent options and clean up your workflow with FlowState Latent Source.

A simple switch to select between:
 - an empty latent
 - input image as a latent
 
Vary the denoise on your sampler to control the amount of style transfer you want from your input image in your generated images.
<p align="center">
  <img width='650' src='https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/imgs/FlowState%20Simple%20Latent.png' alt='FS Simple Latent Image'/>
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
| `vae` | `VAE` | **Variational AutoEncoder (VAE)**: The VAE model used for encoding and decoding images. |
| `input_img` (optional) | `IMAGE` | **Input Image**: An optional image passed from another node, used when `latent_type` is 'Input Image'. |

**Outputs:**
| Name | Type | Description |
| --- | --- | --- |
| `Latent Image` | `LATENT` | The latent image batch. |

<br/>

Video tutorials can be found on YouTube at [🌊 FlowState Creator Suite Playlist](https://www.youtube.com/playlist?list=PLopF-DMGUFkTulZRkSpRmKFcTENKFicws) *(Coming Soon)*

