# 🌊🥗 FlowState Chef Ingredients

The user-friendly input node for the FlowState Chef pipeline.

- Prepare your recipe for the Chef. This node gathers and prepares all the necessary inputs:
- Stitch up to four separate images into a single composition.
- Provide two separate prompts: one for Qwen's direct edit instructions and one for FLUX's final descriptive refinement.
- Bundles everything into a clean chef_ingredients package to simplify your workflow graph.

<p align="center">
<img width='650' src='https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/imgs/FlowState%20Chef.png' alt='FS Chef Image'/>
</p>

**Inputs**
| Name | Type | Description |
| --- | --- | --- |
| `qwen_edit_prompt` | `STRING` | **Qwen Edit Prompt**: A prompt giving direct instructions to the Qwen model on how to edit the image(s). |
| `flux_refinement_prompt` | `STRING` | **FLUX Refinement Prompt**: A prompt describing the final desired scene for the FLUX model to refine. |
| `image` | `IMAGE` | **Image 1**: The first input image. |
| `image_2` (optional) | `IMAGE` | **Image 2**: A second optional input image. |
| `image_3` (optional) | `IMAGE` | **Image 3**: A third optional input image. |
| `image_4` (optional) | `IMAGE` | **Image 4**: A fourth optional input image. |

**Outputs**
| Name | Type | Description |
| --- | --- | --- |
| `chef_ingredients` | `ANY` | A packaged output containing the stitched image and both prompts, ready to be fed into the FlowState Chef node. |

<br/>

Video tutorials can be found on YouTube at [🌊 FlowState Creator Suite Playlist](https://www.youtube.com/playlist?list=PLopF-DMGUFkTulZRkSpRmKFcTENKFicws) *(Coming Soon)*
