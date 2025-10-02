# Project: FlowState Video Creator
# Description: Simple Create Video / Save Video combo.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'\t - 🟢 🎥 Loaded FlowState Video Creator.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FlowState_Node import FlowState_Node


##
# OUTSIDE IMPORTS
##
import time, types

from comfy_api.latest import ComfyExtension, io, ui

from comfy_extras.nodes_video import CreateVideo
from comfy_extras.nodes_video import SaveVideo


##
# NODES
##
class FlowState_VideoCreator(FlowState_Node):
    CATEGORY = 'FlowState Creator Suite/Video'
    DESCRIPTION = 'Simple Create Video / Save Video combo.'
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_TOOLTIPS = ()

    def __init__(self):
        super().__init__('🌊🎥 FlowState Video Creator')
        self.video_params = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'frames': TYPE_FRAMES_IN,
                'fps': TYPE_FPS,
                'format': TYPE_VIDEO_FORMAT,
                'codec': TYPE_VIDEO_CODEC,
                'filename_prefix': TYPE_FILENAME_PREFIX,
            },
            'optional': {
                'audio': TYPE_AUDIO_IN,
            },
            'hidden': {
                'prompt': 'PROMPT',
                'extra_pnginfo': 'EXTRA_PNGINFO'
            }
        }

    def create_video(self):
        self.video = CreateVideo.execute(
            self.video_params['frames'],
            self.video_params['fps'],
            self.video_params['audio']
        )[0]
    
    def save_video(self):
        hidden_inputs = types.SimpleNamespace()
        hidden_inputs.prompt = self.video_params['prompt']
        hidden_inputs.extra_pnginfo = self.video_params['extra_pnginfo']
        SaveVideo.hidden = hidden_inputs

        self.node_output = SaveVideo.execute(
            self.video,
            self.video_params['filename_prefix'],
            self.video_params['format'],
            self.video_params['codec']
        )

    def execute(self, frames, fps, format, codec, filename_prefix, audio=None, prompt=None, extra_pnginfo=None):
        self.print_status([('Preparing video...',)], init=True)

        self.video_params = locals()

        self.create_video()
        self.save_video()

        create_start_time = time.time()

        create_duration, create_mins, create_secs = get_mins_and_secs(create_start_time)

        self.print_status([
            ('Video creation complete.',),
            ('Creation Time', f'{create_mins}m {create_secs}s ({create_duration}s)'),
        ], end=True)

        return self.node_output

