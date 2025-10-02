# Project: FlowState Video Preview
# Description: Simple preview video don't know why this isn't a node.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'\t - 🟢 📺 Loaded FlowState Video Preview.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FlowState_Node import FlowState_Node
from .FlowState_SaveTempVideo import FlowState_SaveTempVideo


##
# OUTSIDE IMPORTS
##
import time, types


##
# NODES
##
class FlowState_VideoPreview(FlowState_Node):
    CATEGORY = 'FlowState Creator Suite/Video'
    DESCRIPTION = "Simple preview video node. Don't know why this isn't a node."
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_TOOLTIPS = ()

    def __init__(self):
        super().__init__('🌊📺 FlowState Video Preview')

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'video': TYPE_VIDEO_IN
            },
            'hidden': {
                'prompt': 'PROMPT',
                'extra_pnginfo': 'EXTRA_PNGINFO'
            }
        }
    
    def setup_namespace(self):
        self.hidden = types.SimpleNamespace()
        self.hidden.prompt = self.video_params['prompt']
        self.hidden.extra_pnginfo = self.video_params['extra_pnginfo']
        FlowState_SaveTempVideo.hidden = self.hidden

    def save_temp_video(self):
        self.node_output = FlowState_SaveTempVideo.execute(
            self.video_params['video']
        )

    def execute(self, video, prompt=None, extra_pnginfo=None):
        self.print_status([('Previewing video...',)], init=True)

        self.video_params = locals()

        self.setup_namespace()
        self.save_temp_video()

        create_start_time = time.time()

        create_duration, create_mins, create_secs = get_mins_and_secs(create_start_time)

        self.print_status([
            ('Video preview complete.',),
            ('Duration', f'{create_mins}m {create_secs}s ({create_duration}s)'),
        ], end=True)

        return self.node_output

