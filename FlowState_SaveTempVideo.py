# Project: FlowState Video Preview
# Description: Simple preview video don't know why this isn't a node.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# OUTSIDE IMPORTS
##
import os
import folder_paths

from datetime import datetime
from comfy_api.input import VideoInput
from comfy_api.util import VideoContainer
from comfy_api.latest import io, ui
from comfy.cli_args import args


class FlowState_SaveTempVideo(io.ComfyNode):
    file_prefix = f'FlowState_VideoPreview'
    format = 'mp4'
    codec = 'h264'

    temp_dir = folder_paths.get_temp_directory()

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='FlowState_SaveTempVideo',
            display_name='FlowState Save Temp Video',
            category='FlowState Creator Suite/Video',
            description='Saves the video previews to your ComfyUI temp directory.',
            inputs=[
                io.Video.Input('video'),
                io.String.Input('filename_prefix'),
                io.Combo.Input('format'),
                io.Combo.Input('codec')
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def save_metadata(cls):
        null_metadata = None

        if args.disable_metadata:
            return null_metadata

        metadata = {}

        if cls.hidden.extra_pnginfo != None:
            metadata.update(cls.hidden.extra_pnginfo)

        if cls.hidden.prompt != None:
            metadata['prompt'] = cls.hidden.prompt

        have_metadata = len(metadata) > 0
        
        return metadata if have_metadata else null_metadata

    @classmethod
    def save_video(cls, video, tmp_dir_path, filename, metadata):
        full_path = os.path.join(tmp_dir_path, filename)

        video.save_to(
            full_path, format=cls.format, codec=cls.codec, metadata=metadata
        )

    @classmethod
    def get_path_parts(cls, width, height):
        tmp_dir_path, filename, counter, subfolder, prefix = folder_paths.get_save_image_path(
            cls.file_prefix, cls.temp_dir, width, height
        )

        return tmp_dir_path, filename, subfolder

    @classmethod
    def execute(cls, video: VideoInput) -> io.NodeOutput:
        width, height = video.get_dimensions()

        tmp_dir_path, filename, subfolder = cls.get_path_parts(width, height)

        saved_metadata = cls.save_metadata()
        video_extension = VideoContainer.get_extension(cls.format)

        timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')[:-3]
        tmp_filename = f'{filename} - {timestamp}.{video_extension}'

        cls.save_video(video, tmp_dir_path, tmp_filename, saved_metadata)

        saved_result = ui.SavedResult(tmp_filename, subfolder, io.FolderType.temp)
        video_preview = ui.PreviewVideo([saved_result])

        return io.NodeOutput(ui=video_preview)

