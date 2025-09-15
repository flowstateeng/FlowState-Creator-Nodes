# Project: FlowState Types
# Description: Global types for all nodes.
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng



##
# FS IMPORTS
##
from .FS_Assets import *
from .FS_Constants import *
from .FS_Types import *
from .FS_Utils import *


##
# OUTSIDE IMPORTS
##
import sys
import nodes


##
# ANY TYPE
##
class AnyType(str):
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

TYPE_ANY = (AnyType('*'), {})



##
# GENERIC INPUT TYPES
##

# NUMERICAL
TYPE_FLOAT = ('FLOAT', {'default': 1, 'min': -sys.float_info.max, 'max': sys.float_info.max, 'step': 0.01})
TYPE_INT = ('INT', {'default': 1, 'min': -sys.maxsize, 'max': sys.maxsize, 'step': 1})

# LOGICAL
TYPE_BOOLEAN = ('BOOLEAN', {'default': True})
TYPE_BOOLEAN_FALSE = ('BOOLEAN', {'default': False})
TYPE_BOOLEAN_TRUE = ('BOOLEAN', {'default': True})
TYPE_BOOLEAN_PARAMS = ('BOOLEAN', {'default': False, 'tooltip': 'Add params to output images.'})
TYPE_BOOLEAN_PROMPT = ('BOOLEAN', {'default': False, 'tooltip': 'Add prompt to output images.'})
TYPE_BOOLEAN_PARAMS_TERM = ('BOOLEAN', {'default': False, 'tooltip': 'Print params to cmd/terminal.'})
TYPE_BOOLEAN_PROMPT_TERM = ('BOOLEAN', {'default': False, 'tooltip': 'Print prompt to cmd/terminal.'})

# STRING
TYPE_STRING_IN = ('STRING', {'default': 'Enter a value.'})
TYPE_STRING_ML = ('STRING', {'multiline': True, 'default': 'Enter a value.'})

# IMAGE
TYPE_IMG_WIDTH = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8, 'tooltip': 'Defines width input image.'})
TYPE_IMG_HEIGHT = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8, 'tooltip': 'Defines height of the input image.'})

# LATENT
TYPE_LATENT_IN = ('LATENT', {'tooltip': 'Input latent image for diffusion process.'})

# SAMPLING
TYPE_POSITIVE_CONDITIONING = ('CONDITIONING', {'tooltip': 'Positive conditioning from encoded text prompt.'})
TYPE_NEGATIVE_CONDITIONING = ('CONDITIONING', {'tooltip': 'Negative conditioning from encoded text prompt. For SD models only. Will not be used for Flux.'})
TYPE_SEED = ('INT', {'default': 4, 'min': -sys.maxsize, 'max': sys.maxsize, 'step': 1, 'tooltip': 'Random noise seed.'})
TYPE_STEPS = ('INT', {'default': 32, 'min': 1, 'max': 10000, 'tooltip': 'Defines the number of steps to take in the sampling process.'})
TYPE_GUIDANCE = ('FLOAT', {'default': 4.0, 'min': 0.0, 'max': 100.0, 'step':0.1, 'round': 0.01, 'tooltip': 'Controls the influence of external guidance (such as prompts or conditions) on the sampling process.'})
TYPE_DENOISE = ('FLOAT', {
    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
    'tooltip': (
        f'Sampler Denoise Amount\n\n'
        f' - The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.\n\n'
    )
})

# MODEL
TYPE_MODEL_IN = ('MODEL', {'tooltip': 'Input model.'})
TYPE_CLIP_IN = ('CLIP', {'tooltip': 'The CLIP model used for encoding the text.'})
TYPE_VAE_IN = ('VAE', {'tooltip': 'The VAE model used for encoding and decoding images.'})
TYPE_CONTROL_NET_IN = ('CONTROL_NET', {'tooltip': 'The Control Net model used to patch your diffusion model.'})
TYPE_LORA_IN = ('LORA', {'tooltip': 'The LoRA used to patch your diffusion model.'})

# MISC
TYPE_JSON_WIDGET = ('JSON', {'forceInput': True})
TYPE_METADATA_RAW = ('METADATA_RAW', {'forceInput': True})



##
# LIST INPUT TYPES
##

# MODELS
TYPE_DIFFUSION_MODELS_LIST = (DIFFUSION_MODELS_LIST(), {'tooltip': 'Diffusion model list.'})
TYPE_CHECKPOINTS_LIST = (CHECKPOINTS_LIST(), {'tooltip': 'Checkpoint list.'})
TYPE_CLIPS_LIST = (CLIPS_LIST(), {'tooltip': 'CLIP/text encoder list.'})
TYPE_VAES_LIST = (VAES_LIST(), {'tooltip': 'VAE list.'})
TYPE_CONTROL_NETS_LIST = (['none'] + CONTROL_NETS_LIST(), {'tooltip': 'Control Net list.'})
TYPE_LORAS_LIST = (['none'] + LORAS_LIST(), {'tooltip': 'LoRA list.'})
TYPE_ALL_MODEL_LISTS = (ALL_MODELS_LIST(), {'tooltip': 'Full diffusion model list.'})

# SAMPLING
TYPE_SAMPLERS = (SAMPLERS_LIST(), {'tool_tip': 'The sampling algorithm(s) used during the diffusion process.'}, )
TYPE_SCHEDULERS = (SCHEDULERS_LIST(), {'tool_tip': 'The scheduling algorithm(s) used during the diffusion process.'}, )

# FILES
TYPE_INPUT_FILES = (sorted(INPUT_FILES()), {'image_upload': True})
TYPE_OUTPUT_FILES = (sorted(OUTPUT_FILES()), {'image_upload': True})



##
# GENERIC OUTPUT TYPES
##
TYPE_MODEL = ('MODEL', )
TYPE_CONDITIONING = ('CONDITIONING', )

TYPE_STRING_OUT = ('STRING', )
TYPE_STRING_OUT_2 = ('STRING', 'STRING', )

TYPE_LATENT = ('LATENT', )
TYPE_IMAGE = ('IMAGE', )



##
# FLOWSTATE CREATOR TYPES
##

# LATENT SOURCE
TYPE_LATENT_BATCH_SIZE = ('INT', {'default': 1, 'min': 1, 'max': 4096, 'tooltip': (
        f'Custom Batch Size\n'
        f'-----------------\n'
        f' - The number of images you want to generate.\n\n'
    )})
TYPE_LATENT_SOURCE_INPUT_TYPE = (['Empty Latent', 'Input Image', 'Uploaded Image'], {
    'tooltip': (
        f'Latent Type\n'
        f'-----------\n'
        f' - Your choice of an empty latent (all zeros) or an image as a latent.\n\n'
    )
})
TYPE_LATENT_SOURCE_RESOLUTION = ([
    'Custom',
    # HORIZONTAL
    '1920x1080 - 16:9',
    '1280x720 - 16:9',
    '1280x768 - 5:3',
    '1280x960 - 4:3',
    '1024x768 - 4:3',
    '2048x512 - 4:1',
    '1152x896 - 9:7',
    '4096x2048 - 2:1',
    '2048x1024 - 2:1',
    '1564x670 - 21:9',
    '2212x948 - 21:9',
    # SQUARE
    '4096x4096 - 1:1',
    '3072x3072 - 1:1',
    '2048x2048 - 1:1',
    '1024x1024 - 1:1',
    '720x720 - 1:1',
    '512x512 - 1:1'
    ], {
    'tooltip': (
        f'Resolution Selector\n'
        f'-------------------\n'
        f' - Select custom to use the entered width & height, or select a resolution.\n\n'
    )
})
TYPE_LATENT_SOURCE_ORIENTATION = (['Horizontal', 'Vertical'], {
    'tooltip': (
        f'Orientaion Selector\n'
        f'-------------------\n'
        f' - Resolutions given in horizontal orientation. Selects vertical to swap.\n\n'
    )
})
TYPE_LATENT_SOURCE_OUT = ('LATENT', )

# ADVANCED SAMPLING
TYPE_ADDED_LINES = ('INT', {'default': 0, 'min': -20, 'max': 50, 'tooltip': 'Add lines to text in image if your prompt is cut off.'})
TYPE_SEED_LIST = ('STRING', {'default': '4', 'tooltip': 'Random noise seed list. If not empty, seed list is used instead of seed.'})
TYPE_STEPS_LIST = ('STRING', {'default': '32', 'tooltip': 'Defines the number of steps to take in the sampling process. Comma-separated list for multiple runs.'})
TYPE_GUIDANCE_LIST = ('STRING', {'default': '4.0', 'tooltip': 'Controls the influence of external guidance (such as prompts or conditions) on the sampling process. Comma-separated list for multiple runs.'})
TYPE_MAX_SHIFT_LIST = ('STRING', {'default': '1.04', 'tooltip': 'Defines the maximum pixel movement for image displacement. Comma-separated list for multiple runs.'})
TYPE_BASE_SHIFT_LIST = ('STRING', {'default': '0.44', 'tooltip': 'Sets the baseline pixel shift applied before variations. Comma-separated list for multiple runs.'})
TYPE_DENOISE_LIST = ('STRING', {
    'default': '1.0',
    'tooltip': (
        f'Sampler Denoise Amount\n\n'
        f' - The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.\n\n'
        f' - Comma-separated list for multiple runs.\n\n'
    )
})
TYPE_LATENT_MULT_LIST = ('STRING', {'default': '1.14', 'tooltip': 'Sets latent multiply factor. Comma-separated list for multiple runs.'})
TYPE_LATENT_MULT = ('FLOAT', {'default': 1.14, 'min': -10.0, 'max': 10.0, 'step': 0.01, 'tooltip': 'Sets latent multiply factor.'})
TYPE_FONT_SIZE = ('INT', {'default': 42, 'min': 16, 'max': 96, 'step': 1, 'tooltip': 'Defines burned-in font size.'})

# STYLE MODELS
style_list = ['none', 'lora', 'lora & none']
TYPE_STYLE_TYPE = (style_list, {
    'tooltip': (
        f'Unified Styler Type\n\n'
        f' - none: runs your original model as is.\n\n'
        f' - control net: applies selected control net with specified settings.\n\n'
        f' - lora: applies selected control net with specified settings.\n\n'
        f' - control net & lora: 2 runs - one with control net & one with lora.\n\n'
        f' - control net & none: 2 runs - one with control net & one with no style models applied.\n\n'
        f' - lora & none: 2 runs - one with lora & one with no style models applied.\n\n'
        f' - all three: 3 runs - one with control net, one with lora & one with no style models applied.\n\n'
    )
})
TYPE_CONTROL_NET_STRNGTH = ('STRING', {'default': '1.0',
    'tooltip': (
        f'Control Net Strength\n\n'
        f' - default: 1.0\n\n'
        f' - min: 0.0\n\n'
        f' - max: 10.0\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for strengths.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
TYPE_CONTROL_NET_START = ('STRING', {'default': '0.0',
    'tooltip': (
        f'Control Net Start\n\n'
        f' - default: 0.0\n\n'
        f' - min: 0.0\n\n'
        f' - max: 1.0\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for starts.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
TYPE_CONTROL_NET_END = ('STRING', {'default': '1.0',
    'tooltip': (
        f'Control Net End\n\n'
        f' - default: 1.0\n\n'
        f' - min: 0.0\n\n'
        f' - max: 1.0\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for ends.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
TYPE_CANNY_THRESHOLD_LOW = ('STRING', {'default': '0.4',
    'tooltip': (
        f'Canny Threshold Low\n\n'
        f' - default: 0.4\n\n'
        f' - min: 0.01\n\n'
        f' - max: 0.99\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for low thresholds.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
TYPE_CANNY_THRESHOLD_HIGH = ('STRING', {'default': '0.8',
    'tooltip': (
        f'Canny Threshold High\n\n'
        f' - default: 0.8\n\n'
        f' - min: 0.01\n\n'
        f' - max: 0.99\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for high thresholds.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
TYPE_LORA_STRENGTH = ('STRING', {'default': '1.0',
    'tooltip': (
        f'LoRA Strength\n\n'
        f' - default: 1.0\n\n'
        f' - min: -100.0\n\n'
        f' - max: 100.0\n\n'
        f' * if using multiple loras, use the following comma-separated list format for strengths.\n\n'
        f' - 1 lora: 0.4\n\n'
        f' - 2 loras: 0.4, 0.5\n\n'
        f' - 3 loras: 0.4, 0.5, 0.6\n\n'
    )
})
TYPE_LORA_CLIP_STRENGTH = ('STRING', {'default': '1.0',
    'tooltip': (
        f'LoRA CLIP Strength\n\n'
        f' - default: 1.0\n\n'
        f' - min: -100.0\n\n'
        f' - max: 100.0\n\n'
        f' * if using multiple loras, use the following comma-separated list format for CLIP strengths.\n\n'
        f' - 1 lora: 0.4\n\n'
        f' - 2 loras: 0.4, 0.5\n\n'
        f' - 3 loras: 0.4, 0.5, 0.6\n\n'
    )
})



