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
    'FlowState_LatentSelector': FlowState_LatentSelector,
    # 'FlowState_FluxSampler': FlowState_FluxSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'FlowState_LatentSelector': '🌊 FlowState Latent Selector',
    # 'FlowState_FluxSampler': '🌊 FlowState Flux Sampler',
}


##
# SYSTEM STATUS
##
for fs_node in NODE_CLASS_MAPPINGS:
    print(f'\t - ✅ {fs_node}: {NODE_DISPLAY_NAME_MAPPINGS[fs_node]}')

print(f'\t - ✅ Mappings Loaded.')
