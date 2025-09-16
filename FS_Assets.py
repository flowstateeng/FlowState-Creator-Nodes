# Project: FlowState Assets
# Description: Paths to assets needed by nodes.
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# OUTSIDE IMPORTS
##
from .FS_Utils import *



##
# OUTSIDE IMPORTS
##
import os
import comfy
import folder_paths


##
# ASSETS
##
WEB_DIRECTORY = './web'
FONT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts/ShareTechMono-Regular.ttf')


DIFFUSION_MODELS_LIST = lambda: folder_paths.get_filename_list('unet') + folder_paths.get_filename_list('diffusion_models')
CHECKPOINTS_LIST = lambda: folder_paths.get_filename_list('checkpoints')
CLIPS_LIST = lambda: folder_paths.get_filename_list('clip')
VAES_LIST = lambda: get_vae_list()

CONTROL_NETS_LIST = lambda: folder_paths.get_filename_list('controlnet')
LORAS_LIST = lambda: folder_paths.get_filename_list('loras')

SAMPLERS_LIST = lambda: comfy.samplers.KSampler.SAMPLERS
SCHEDULERS_LIST = lambda: comfy.samplers.KSampler.SCHEDULERS

ALL_MODELS_LIST = lambda: sorted(set(DIFFUSION_MODELS_LIST() + CHECKPOINTS_LIST()))

INPUT_FILES = lambda: get_input_files()
OUTPUT_FILES = lambda: get_output_files()

