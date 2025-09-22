# Project: FlowState Types
# Description: Global types for all nodes.
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng



##
# FS IMPORTS
##
from .FS_Assets import *
from .FS_Constants import *


##
# OUTSIDE IMPORTS
##
import sys
import nodes



# -----------------------------------------------------------
#                    BEGIN GENERIC TYPES
# -----------------------------------------------------------

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
TYPE_IMG_WIDTH = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 1, 'tooltip': (
    f' Width\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Defines the width of the image.\n\n'
)})
TYPE_IMG_HEIGHT = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 1, 'tooltip': (
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
TYPE_PROMPT_POSITIVE = ('STRING', {'multiline': True, 'default': '✅ Describe the image you want the model to create.', 'tooltip': (
    f' Positive Prompt\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - ✅ Describe the image you want the model to create.\n\n'
)})
TYPE_PROMPT_NEGATIVE = ('STRING', {'multiline': True, 'default': '⛔ Describe what you do not want to see in the image.', 'tooltip': (
    f' Positive Prompt\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - ⛔ Describe what you do not want to see in the image.\n\n'
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
TYPE_WEIGHT_DTYPE = (['default', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {'tooltip': (
    f' Weight Datatype (DType)\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The data type to be used for your models weights.\n\n'
)})

# STYLE
TYPE_LORA_IN = ('LORA', {'tooltip': (
    f' Low Rank Adaptation Model (LoRA)\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The LoRA used to patch your diffusion model.\n\n'
)})
TYPE_LORA_STRENGTH = ('FLOAT', {
    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
    'tooltip': (
        f' Low Rank Adaptation Model (LoRA)\n {"-" * TOOLTIP_UNDERLINE}\n'
        f' - The LoRA used to patch your diffusion model.\n\n'
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

TYPE_CONTROL_NETS_LIST = lambda: (['disabled'] + CONTROL_NETS_LIST(), {'tooltip': (
    f' Control Net List\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - List of available Control Nets.\n'
    f' - Used to transfer structure of an input image to a generated output image.\n\n'
)})
TYPE_LORAS_LIST = lambda: (['disabled'] + LORAS_LIST(), {'tooltip': (
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



# -----------------------------------------------------------
#                    BEGIN FLOWSTATE TYPES
# -----------------------------------------------------------


##
# FLOWSTATE CREATOR LABELS
##

def pad_label(label):
    width = 100
    len_label = len(label)
    num_spaces = width - len_label
    left_side = num_spaces // 2 - 10
    right_side = num_spaces - left_side
    padded_label = ' ' * left_side + '--- ' + label + ' ---' + ' ' * right_side
    return padded_label

# GENERIC LABELS
TYPE_FLOWSTATE_LABEL_MODEL = ('STRING', {'default': pad_label('🤖 Model Settings'), 'tooltip': (
    f' Label\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - This field is not functional. It is just a label for the group of settings below.\n\n'
)})

TYPE_FLOWSTATE_LABEL_AUG = ('STRING', {'default': pad_label('🔥 Augmentation Settings'), 'tooltip': (
    f' Label\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - This field is not functional. It is just a label for the group of settings below.\n\n'
)})

TYPE_FLOWSTATE_LABEL_ENCODER = ('STRING', {'default': pad_label('🔣 Encoder Settings'), 'tooltip': (
    f' Label\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - This field is not functional. It is just a label for the group of settings below.\n\n'
)})

TYPE_FLOWSTATE_LABEL_IMAGE = ('STRING', {'default': pad_label('🖼️ Image Settings'), 'tooltip': (
    f' Label\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - This field is not functional. It is just a label for the group of settings below.\n\n'
)})

TYPE_FLOWSTATE_LABEL_VIDEO = ('STRING', {'default': pad_label('🎥 Video Settings'), 'tooltip': (
    f' Label\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - This field is not functional. It is just a label for the group of settings below.\n\n'
)})

TYPE_FLOWSTATE_LABEL_SAMPLING = ('STRING', {'default': pad_label('🧪 Sampling Settings'), 'tooltip': (
    f' Label\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - This field is not functional. It is just a label for the group of settings below.\n\n'
)})

TYPE_FLOWSTATE_LABEL_PROMPT = ('STRING', {'default': pad_label('📝 Prompt(s)'), 'tooltip': (
    f' Label\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - This field is not functional. It is just a label for the group of settings below.\n\n'
)})


##
# FLOWSTATE CREATOR GENERIC TYPES
##


# SAGE ATTENTION
enabled_sage_modes = [
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++"
] if SAGE_ATTENTION_INSTALLED else ['disabled']

TYPE_SAGE_ATTENTION_MODE = (enabled_sage_modes, {'tooltip': (
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


# VIDEO
TYPE_NUM_VIDEO_FRAMES = ('INT', {'default': 48, 'min': 1, 'max': nodes.MAX_RESOLUTION, 'step': 1, 'tooltip': (
    f' Number of Video Frames\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The number of frames you want in your final video.\n\n'
)})


##
# FLOWSTATE VIDEO CREATOR
##
TYPE_FPS = ('INT', {'default': 12, 'min': 1, 'max': 120, 'tooltip': (
    f' Frames Per Second\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The number of frames per second in the created video.\n\n'
)})
TYPE_FRAMES_IN = ('IMAGE', {'tooltip': (
    f' Video Frames\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The frames used to create the video.\n\n'
)})
TYPE_AUDIO_IN = ('AUDIO', {'tooltip': (
    f' Video Audio\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Optional audio to be added to the video.\n\n'
)})
TYPE_FILENAME_PREFIX = ('STRING', {'default': 'video/ComyUI', 'tooltip': (
    f' Filename Prefix\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - The prefix for the file to save.\n'
    f' - This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.\n\n'
)})
TYPE_VIDEO_FORMAT = (['mp4', 'auto'], {
    'tooltip': (
        f' Video Format\n {"-" * TOOLTIP_UNDERLINE}\n'
        f' - The format to save the video as.\n\n'
    )
})
TYPE_VIDEO_CODEC = (['h264', 'auto'], {
    'tooltip': (
        f' Video Codec\n {"-" * TOOLTIP_UNDERLINE}\n'
        f' - The codec to use for the video.\n\n'
    )
})


##
# FLOWSTATE SIMPLE LATENT
##
TYPE_SIMPLE_LATENT_INPUT_TYPE = (['Empty Latent', 'Input Image'], {
    'tooltip': (
        f' Latent Type\n {"-" * TOOLTIP_UNDERLINE}\n'
        f' - Your choice of an empty latent (all zeros) or an image as a latent.\n\n'
    )
})


##
# FLOWSTATE LATENT SOURCE
##
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
        f' - Select "Custom" to use the entered custom_width & custom_height.\n'
        f' - Select a preset resolution & orientation.\n\n'
    )
})
TYPE_LATENT_SOURCE_ORIENTATION = (['Horizontal', 'Vertical'], {
    'tooltip': (
        f' Orientaion Selector\n {"-" * TOOLTIP_UNDERLINE}\n'
        f' - Resolutions given in horizontal orientation. Select vertical to swap resolution aspect ratio.\n\n'
    )
})
TYPE_LATENT_SOURCE_OUT = ('LATENT',)


##
# FLOWSTATE FLUX ENGINE
##
TYPE_PROMPT_FLUX_ENGINE = ('STRING', {'multiline': True, 'default': '✅ Describe the image you want Flux to create.', 'tooltip': (
    f' Positive Prompt\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - ✅ Describe the image you want Flux to create.\n\n'
)})
TYPE_FLUX_ENGINE_OUT = ('MODEL', 'CLIP', 'VAE', 'IMAGE', 'LATENT')


##
# FLOWSTATE WAN STUDIO
##
TYPE_PROMPT_WAN_STUDIO_POSITIVE = ('STRING', {'multiline': True, 'default': '✅ Describe the video you want WAN to create.', 'tooltip': (
    f' Positive Prompt\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - ✅ Describe the video you want WAN to create.\n\n'
)})
TYPE_PROMPT_WAN_STUDIO_NEGATIVE = ('STRING', {'multiline': True, 'default': '⛔ Describe what you do not want to see in the video.', 'tooltip': (
    f' Positive Prompt\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - ⛔ Describe what you do not want to see in the video.\n\n'
)})
TYPE_WAN_STUDIO_STARTING_FRAME = ('IMAGE', {'tooltip': (
    f' Starting Frame\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Optionally, select an input image to use as the starting frame.\n\n'
)})
TYPE_WAN_STUDIO_RESOLUTION = ([
    'Custom',
    'Use Starting Frame Resolution',
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
        f' - Select "Custom" to use the entered custom_width & custom_height.\n'
        f' - Select "Use Starting Frame Resolution" to use the resolution of the input image.\n'
        f' - Select a preset resolution & orientation.\n\n'
    )
})
TYPE_WAN_CLIP_VISION = ('CLIP_VISION', {'tooltip': (
    f' CLIP Vision Output\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Optionally, use a CLIP Vision model.\n\n'
)})
TYPE_WAN_STUDIO_OUT = ('IMAGE', 'LATENT')


##
# FLOWSTATE QUICK EDIT
##
TYPE_PROMPT_QUICK_EDIT_CHANGES = ('STRING', {'multiline': True, 'default': 'Describe the edits you want Qwen to make.', 'tooltip': (
    f' Positive Prompt\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Describe the edits you want Qwen to make.\n\n'
)})
TYPE_PROMPT_QUICK_EDIT_REFINE = ('STRING', {'multiline': True, 'default': 'Describe the new image after the edits are made.', 'tooltip': (
    f' Positive Prompt\n {"-" * TOOLTIP_UNDERLINE}\n'
    f' - Describe the new image after the edits are made.\n\n'
)})
TYPE_QUICK_EDIT_OUT = ('MODEL', 'CLIP', 'VAE', 'IMAGE', 'LATENT')

