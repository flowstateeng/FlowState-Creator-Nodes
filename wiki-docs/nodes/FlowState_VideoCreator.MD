# 🌊🎥 FlowState Video Creator

A simple Create Video / Save Video combo node.

I don't know why this isn't already a node.

<p align="center">
  <img width='650' src='https://github.com/flowstateeng/FlowState-Creator-Nodes/blob/main/imgs/FlowState%20WAN%20Studio.png' alt='FS Video Creator'/>
</p>

**Inputs:**
| Name | Type | Description |
| --- | --- | --- |
| `frames` | `IMAGE` | **Frames**: The frames used to create the video. |
| `fps` | `INT` | **Frames Per Second (FPS)**: The number of frames per second in the created video. |
| `format` | `string` | **Format**: The format to save the video as. |
| `codec` | `STRING` | **Codec**: The codec to use for the video. |
| `filename_prefix` | `STRING` | **Filename Prefix**: The prefix for the file to save. |
| `audio` (optional) | `AUDIO` | **Audio**: Optional audio to be added to the video. |

**Outputs:**
| Name | Type | Description |
| --- | --- | --- |
| `None` | `NONE` | There is no output for this node. |

---

*⚠️ This functionality will be built into all video generation nodes going forward, but this node remains in case you want it for something else.*

<br/>

Video tutorials can be found on YouTube at [🌊 FlowState Creator Suite Playlist](https://www.youtube.com/playlist?list=PLopF-DMGUFkTulZRkSpRmKFcTENKFicws) *(Coming Soon)*
