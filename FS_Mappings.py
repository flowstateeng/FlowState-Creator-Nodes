# Project: FlowState Node Mappings
# Description: Node mappings for ComfyUI registry.
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# FS IMPORTS
##
from .FS_Nodes import *


##
# SYSTEM STATUS
##
print(f'  3. 💾 Loading node name mappings...')


##
# MAPPINGS
##
NODE_CLASS_MAPPINGS = {
    'FlowState_SimpleLatent': FlowState_SimpleLatent,
    'FlowState_VideoCreator': FlowState_VideoCreator,
    'FlowState_LatentSource': FlowState_LatentSource,
    'FlowState_FluxEngine': FlowState_FluxEngine,
    'FlowState_WANStudio': FlowState_WANStudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'FlowState_VideoCreator': '🌊🎥 FlowState Video Creator',
    'FlowState_SimpleLatent': '🌊👌 FlowState Simple Latent',
    'FlowState_LatentSource': '🌊🌱 FlowState Latent Source',
    'FlowState_FluxEngine': '🌊🚒 FlowState Flux Engine',
    'FlowState_WANStudio': '🌊🎬 FlowState WAN Studio',
    # 'FlowState_QuickEdit': '🌊 FlowState Quick Edit',
    # 'FlowState_AssetForge': '🌊 FlowState Asset Forge',
}


##
# SYSTEM STATUS
##
for fs_node in NODE_CLASS_MAPPINGS:
    print(f'\t - 🟢 {fs_node}: {NODE_DISPLAY_NAME_MAPPINGS[fs_node]}')

print(f'\t   - ✅ Mappings Loaded.')
