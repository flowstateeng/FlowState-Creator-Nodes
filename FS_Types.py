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
TYPE_FLOAT = ('FLOAT', {'default': 1, 'min': -sys.float_info.max, 'max': sys.float_info.max, 'step': 0.01, 'tooltip': (
    f' Float\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - A floating point number.\n\n'
)})
TYPE_INT = ('INT', {'default': 1, 'min': -sys.maxsize, 'max': sys.maxsize, 'step': 1, 'tooltip': (
    f' Integer\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - An integer number.\n\n'
)})

# LOGICAL
TYPE_BOOLEAN = ('BOOLEAN', {'default': True, 'tooltip': (
    f' Boolean\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - A logical operator (True/False).\n\n'
)})
TYPE_BOOLEAN_FALSE = ('BOOLEAN', {'default': False, 'tooltip': (
    f' Boolean (False)\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - A logical operator (True/False; default False).\n\n'
)})
TYPE_BOOLEAN_TRUE = ('BOOLEAN', {'default': True, 'tooltip': (
    f' Boolean (True)\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - A logical operator (True/False; default True).\n\n'
)})

# STRING
TYPE_STRING_IN = ('STRING', {'default': 'Enter a value.', 'tooltip': (
    f' String\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - String input (text).\n\n'
)})
TYPE_STRING_ML = ('STRING', {'multiline': True, 'default': 'Enter a value.', 'tooltip': (
    f' Multiline String\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Multiline string input (text).\n\n'
)})

# IMAGE
TYPE_IMG_WIDTH = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8, 'tooltip': (
    f' Width\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Defines the width of the image.\n\n'
)})
TYPE_IMG_HEIGHT = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8, 'tooltip': (
    f' Height\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Defines the height of the image.\n\n'
)})

# LATENT
TYPE_LATENT_IN = ('LATENT', {'tooltip': (
    f' Latent Image\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Input latent image for diffusion sampling.\n\n'
)})

# SAMPLING
TYPE_POSITIVE_CONDITIONING = ('CONDITIONING', {'tooltip': (
    f' Positive Conditioning\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Positive conditioning from encoded text prompt.\n\n'
)})
TYPE_NEGATIVE_CONDITIONING = ('CONDITIONING', {'tooltip': (
    f' Negative Conditioning\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Negative conditioning from encoded text prompt.\n\n'
)})
TYPE_SEED = ('INT', {'default': 32, 'min': -sys.maxsize, 'max': sys.maxsize, 'step': 1, 'tooltip': (
    f' Seed\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Seed used to generate inital random noise.\n\n'
)})
TYPE_STEPS = ('INT', {'default': 32, 'min': 1, 'max': 10000, 'tooltip': (
    f' Steps\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Defines the number of steps to take in the sampling process.\n\n'
)})
TYPE_GUIDANCE = ('FLOAT', {'default': 3.2, 'min': 0.0, 'max': 100.0, 'step':0.1, 'round': 0.01, 'tooltip': (
    f' Guidance\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Defines the number of steps to take in the sampling process.\n\n'
)})
TYPE_DENOISE = ('FLOAT', {
    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
    'tooltip': (
        f' Sampler Denoise Amount\n {"-" * TOOLTIP_UNDERLINE}\n'
        f' - The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.\n\n'
    )
})
TYPE_PROMPT_POSITIVE = ('STRING', {'multiline': True, 'default': 'Enter your positive prompt.', 'tooltip': (
    f' Positive Prompt\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Positive text prompt describing your desired output.\n\n'
)})

# MODEL
TYPE_MODEL_IN = ('MODEL', {'tooltip': (
    f' Input Model\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Diffusion model to be used in sampling.\n\n'
)})
TYPE_CLIP_IN = ('CLIP', {'tooltip': (
    f' CLIP / Text Encoder Model\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The CLIP / Text Encoder model used for encoding the text.\n\n'
)})
TYPE_VAE_IN = ('VAE', {'tooltip': (
    f' Variational AutoEncoder (VAE)\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The VAE model used for encoding and decoding images.\n\n'
)})
TYPE_CONTROL_NET_IN = ('CONTROL_NET', {'tooltip': (
    f' Control Net\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The Control Net model used to patch your diffusion model.\n\n'
)})
TYPE_LORA_IN = ('LORA', {'tooltip': (
    f' Low Rank Adaptation Model (LoRA)\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The LoRA used to patch your diffusion model.\n\n'
)})
TYPE_WEIGHT_DTYPE = (['default', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {'tooltip': (
    f' Weight Datatype (DType)\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The data type to be used for your models weights.\n\n'
)})

# MISC
TYPE_JSON_WIDGET = ('JSON', {'forceInput': True})
TYPE_METADATA_RAW = ('METADATA_RAW', {'forceInput': True})



##
# LIST INPUT TYPES
##

# MODELS
TYPE_DIFFUSION_MODELS_LIST = lambda: (DIFFUSION_MODELS_LIST(), {'tooltip': (
    f' Diffusion Model List\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of available diffusion models.\n\n'
)})
TYPE_CHECKPOINTS_LIST = lambda: (CHECKPOINTS_LIST(), {'tooltip': (
    f' Checkpoint List\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of available model checkpoints.\n\n'
)})
TYPE_CLIPS_LIST = lambda: (CLIPS_LIST(), {'tooltip': (
    f' CLIP / Text Encoder List\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of available Text Encoders and CLIP models.\n'
    f' - Used to convert your text prompts into semantic attention vectors (i.e., numbers) that the model can process.\n'
    f' - Contrastive Language-Image Pre-training (CLIP)\n\n'
)})
TYPE_VAES_LIST = lambda: (VAES_LIST(), {'tooltip': (
    f' VAE List\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of available Variational Autoencoders (VAE).\n'
    f' - Used to encode and decode images.\n\n'
)})
TYPE_CONTROL_NETS_LIST = lambda: (['none'] + CONTROL_NETS_LIST(), {'tooltip': (
    f' Control Net List\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of available Control Nets.\n'
    f' - Used to transfer structure of an input image to a generated output image.\n\n'
)})
TYPE_LORAS_LIST = lambda: (['none'] + LORAS_LIST(), {'tooltip': (
    f' LoRA List\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of available Low-Rank Adaptation models.\n'
    f' - Used to transfer a pre-trained style (cyberpunk, anime, photorealism, disney, etc.) to a generated output image.\n\n'
)})
TYPE_ALL_MODEL_LISTS = lambda: (ALL_MODELS_LIST(), {'tooltip': (
    f' Full Diffusion Model List\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of all available Diffusion Models (diffusion_models, checkpoints & unets folders).\n\n'
)})

# SAMPLING
TYPE_SAMPLERS = lambda: (SAMPLERS_LIST(), {'tooltip': (
    f' Sampling Algorithm\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of available Sampling Algorithms.\n'
    f' - Used to control the noise removal during the sampling process.\n\n'
)})
TYPE_SCHEDULERS = lambda: (SCHEDULERS_LIST(), {'tooltip': (
    f' Scheduling Algorithm\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of available Scheduling Algorithms.\n'
    f' - Used to control the denoising steps during the sampling process.\n\n'
)})

# FILES
TYPE_INPUT_FILES = lambda: (sorted(INPUT_FILES()), {'image_upload': True})
TYPE_OUTPUT_FILES = lambda: (sorted(OUTPUT_FILES()), {'image_upload': True})



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

# SAGE ATTENTION
TYPE_SAGE_ATTENTION_MODE = (['disabled'], {'tooltip': (
    f' Sage Attention Mode\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The type of Sage Attention to use.\n'
    f' - This field will only show as "disabled" if you do not have the capability to run Sage Attention.\n\n'
)})

# MODEL
TYPE_MODEL_FILE_TYPE = (['solo_model', 'checkpoint'], {'tooltip': (
    f' Model File Type\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The type of model file to load.\n'
    f' - Checkpoints (typically for fp8 models) contain the CLIP & VAE.\n'
    f' - If using a checkpoint, then the weight_dtype, clip_1_name, clip_2_name & vae_name fields will ignored.\n\n'
)})

# LATENT SOURCE
TYPE_LATENT_BATCH_SIZE = ('INT', {'default': 1, 'min': 1, 'max': 4096, 'tooltip': (
        f' Custom Batch Size\n {"-" * TOOLTIP_UNDERLINE}\n'
        f' - The number of images you want to generate.\n\n'
    )})
TYPE_LATENT_SOURCE_INPUT_TYPE = (['Empty Latent', 'Input Image', 'Uploaded Image'], {
    'tooltip': (
        f' Latent Type\n {"-" * TOOLTIP_UNDERLINE}\n'
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
        f' Resolution Selector\n {"-" * TOOLTIP_UNDERLINE}\n'
        f' - Select custom to use the entered width & height, or select a resolution.\n\n'
    )
})
TYPE_LATENT_SOURCE_ORIENTATION = (['Horizontal', 'Vertical'], {
    'tooltip': (
        f' Orientaion Selector\n {"-" * TOOLTIP_UNDERLINE}\n'
        f' - Resolutions given in horizontal orientation. Select vertical to swap resolution aspect ratio.\n\n'
    )
})
TYPE_LATENT_SOURCE_OUT = ('LATENT', )

# FLUX ENGINE
TYPE_FLUX_ENGINE_OUT = ('MODEL', 'CLIP', 'VAE', 'IMAGE', 'LATENT', )


# --- TO DO ---
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



