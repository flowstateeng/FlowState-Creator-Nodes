# Project: FlowState FluxSampler
# Description: One sampler to rule them all.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'    - Loaded Unified Sampler.')


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
import time, copy, itertools, math

import torch
import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import warnings

import comfy.utils
import comfy.sd

from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from comfy_extras.nodes_custom_sampler import BasicGuider
from comfy_extras.nodes_custom_sampler import KSamplerSelect
from comfy_extras.nodes_custom_sampler import BasicScheduler
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_flux import FluxGuidance


from nodes import EmptyLatentImage
from nodes import CLIPTextEncode
from nodes import VAEEncode
from nodes import VAEDecode



from node_helpers import conditioning_set_values

from nodes import common_ksampler


warnings.filterwarnings('ignore', message='clean_up_tokenization_spaces')
warnings.filterwarnings('ignore', message='Torch was not compiled with flash attention')
warnings.filterwarnings('ignore', category=FutureWarning)


##
# NODES
##
class FlowState_FluxSampler:
    CATEGORY = 'FlowState/sampler'
    DESCRIPTION = ('Simple sampler for Flux models.')
    FUNCTION = 'execute'
    RETURN_TYPES = TYPE_IMAGE
    RETURN_NAMES = ('images')
    OUTPUT_TOOLTIPS = (
        'The image batch.',
        'The latent batch.',
        'The parameters used for the image batch.',
    )

    def __init__(self):
        self.prev_params = []
        self.last_latent_batch = None
        self.last_img_batch = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'model': TYPE_DIFFUSION_MODELS_LIST(),
                'clip_1': TYPE_CLIPS_LIST(),
                'clip_2': TYPE_CLIPS_LIST(),
                'vae': TYPE_VAES_LIST(),
                'seed': TYPE_SEED,
                'sampling_algorithm': TYPE_SAMPLERS(),
                'scheduling_algorithm': TYPE_SCHEDULERS(),
                'guidance': TYPE_GUIDANCE,
                'steps': TYPE_STEPS,
                'denoise': TYPE_DENOISE,
                'prompt': TYPE_STRING_ML,
                # 'add_params': BOOLEAN_PARAMS,
                # 'add_prompt': BOOLEAN_PROMPT,
                # 'show_params_in_terminal': BOOLEAN_PARAMS_TERM,
                # 'show_prompt_in_terminal': BOOLEAN_PROMPT_TERM,
                # 'font_size': FONT_SIZE,
            },
            # 'optional': {
            #     'added_lines': ADDED_LINES,
            #     'seed_str_list': SEED_LIST,
            # }
        }

    def add_params(self, img_batch, params, width, height, font_size=42, added_lines=0):
        params_copy = copy.deepcopy(params)

        if 'add_params' in params_copy:
            del params_copy['add_params']

        if 'add_prompt' in params_copy:
            del params_copy['add_prompt']

        using_prompt = 'prompt' in params_copy
        using_params = len(params_copy) > 1
        using_params_but_not_prompt = using_params and not using_prompt
        using_prompt_but_not_params = using_prompt and not using_params
        num_lines = 7 if using_params_but_not_prompt else (7 if using_prompt_but_not_params else 14)

        print(
            f'\nFlowState Flux Sampler - Adding Params.'
            f'\n  - Adding Prompt: {using_prompt}'
        )

        start_time = time.time()

        # Loop over the batch of images
        updated_img_batch_list = []
        for img in img_batch:
            image_np = img.numpy()
            img_min = image_np.min()
            img_max = image_np.max()

            image_norm = (image_np - img_min) / (img_max - img_min) * 255
            image_int = image_norm.astype(np.uint8)
            image = Image.fromarray(image_int)

            # Add text
            font = ImageFont.truetype(FONT_PATH, font_size)
            bbox = font.getbbox('A')
            char_width = bbox[2] - bbox[0]
            line_height = font.getmetrics()[1]
            max_line_len = width // char_width - 2

            # Split parameters into lines of text
            wrapped_text = self.split_params(params_copy, max_line_len - 1)

            # Create a new image with space for text at the bottom
            params_bar_height = (math.ceil(num_lines / 4) * 4 + added_lines) * font_size
            updated_img = Image.new('RGB', (width, height + params_bar_height), (0, 0, 0))
            updated_img.paste(image, (0, 0))

            # Draw text on the image
            draw = ImageDraw.Draw(updated_img)
            y_text = height + font_size // 2

            for line in wrapped_text:
                draw.text((char_width, y_text), line, font=font, fill=(255, 255, 255))
                y_text += line_height + font_size

            # Append updated image to the batch list
            updated_img_batch_list.append(updated_img)

        # Convert the list of PIL images back to a 4D tensor and permute to (n_imgs, height + bar_height, width, 3)
        updated_img_batch_tensor = torch.stack([F.to_tensor(img).permute(1, 2, 0) for img in updated_img_batch_list])

        params_duration = time.time() - start_time
        params_mins = int(params_duration // 60)
        params_secs = int(params_duration - params_mins * 60)

        print(f'  - Complete. Params Duration: {params_mins}m {params_secs}s\n')

        # Return the updated 4D tensor
        return updated_img_batch_tensor

    def check_params(self, params, params_num):
        num_prev_params = len(self.prev_params)
        have_prev_params = num_prev_params > 0
        no_prev_params = not have_prev_params

        adding_params = params['add_params'] == True
        adding_prompt = params['add_prompt'] == True

        in_range = -num_prev_params <= params_num < num_prev_params
        more_imgs = not in_range

        first_batch = no_prev_params or more_imgs or self.last_latent_batch == None or self.last_img_batch == None

        actions = []

        if first_batch:
            print(f'  - First Run.')
            actions.append('run')
            if adding_params: actions.append('add_params')
            if adding_prompt: actions.append('add_prompt')
            self.reset()
            return actions, params

        new_params_stashed_copy = copy.deepcopy(params)
        new_params_working_copy = copy.deepcopy(params)

        prev_params_stashed_copy = copy.deepcopy(self.prev_params[params_num])
        prev_params_working_copy = copy.deepcopy(self.prev_params[params_num])

        for k, v in prev_params_stashed_copy.items():
            if k.startswith('llm_'):
                del prev_params_working_copy[k]

        for k, v in prev_params_stashed_copy.items():
            if k.startswith('llm_') and k in new_params_working_copy:
                del new_params_working_copy[k]

        del new_params_working_copy['add_params']
        del new_params_working_copy['add_prompt']

        del prev_params_working_copy['add_params']
        del prev_params_working_copy['add_prompt']
        del prev_params_working_copy['sampling_duration']

        prev_params_added = prev_params_stashed_copy['add_params'] == True
        prev_prompt_added = prev_params_stashed_copy['add_prompt'] == True
        prev_params_not_added = not prev_params_added
        prev_prompt_not_added = not prev_prompt_added

        new_params_added = params['add_params'] == True
        new_prompt_added = params['add_prompt'] == True
        new_params_not_added = not new_params_added
        new_prompt_not_added = not new_prompt_added

        running = prev_params_working_copy != new_params_working_copy
        not_running = not running

        if running:
            actions.append('run')

        if not_running:
            new_params_working_copy['sampling_duration'] = prev_params_stashed_copy['sampling_duration']
            new_params_stashed_copy['sampling_duration'] = prev_params_stashed_copy['sampling_duration']

        if new_params_added and prev_params_not_added:
            actions.append('add_params')

        if new_params_not_added and prev_params_added:
            actions.append('remove_params')

        if new_params_added and prev_params_added:
            actions.append('keep_params')

        if new_prompt_added and prev_prompt_not_added:
            actions.append('add_prompt')

        if new_prompt_not_added and prev_prompt_added:
            actions.append('remove_prompt')

        if new_prompt_added and prev_prompt_added:
            actions.append('keep_prompt')


        no_actions_taken = len(actions) == 0
        if no_actions_taken:
            return None, new_params_working_copy

        return actions, new_params_stashed_copy

    def sample(self, model, clip_1, clip_2, vae, seed, sampling_algorithm, scheduling_algorithm, guidance, steps, denoise, prompt):

        print(
            f'\n\nFlowState Flux Sampler'
            # f'\n  - Preparing run: ({run_num}/{num_runs})'
        )

        start_time = time.time()

        width = latent_img['samples'].shape[3] * 8
        height = latent_img['samples'].shape[2] * 8

        params = {
            'model_type': model_type,
            'seed': seed,
            'width': width,
            'height': height,
            'sampler': sampling_algorithm,
            'scheduler': scheduling_algorithm,
            'steps': steps,
            'guidance': guidance,
            'max_shift': max_shift,
            'base_shift': base_shift,
            'denoise': denoise,
            'multiplier': multiplier,
            'add_params': add_params,
            'add_prompt': add_prompt,
            'prompt': {
                'positive': positive_prompt,
                'negative': negative_prompt if model_type == 'SD' else None
            }
        }

        if show_params_in_terminal:
            print(f'  - Params:\n    - {log_params}\n')

        if show_prompt_in_terminal:
            print(f'  - Prompt:\n    - {params["prompt"]}\n')


        randnoise = Noise_RandomNoise(seed)
        conditioning = conditioning_set_values(positive_conditioning, {'guidance': guidance})
        guider = BasicGuider().get_guider(model, conditioning)[0]
        sampler = comfy.samplers.sampler_object(sampling_algorithm)
        sigmas = BasicScheduler().get_sigmas(model, scheduling_algorithm, steps, denoise)[0]
        flux_out = SamplerCustomAdvanced().sample(randnoise, guider, sampler, sigmas, latent_batch)[1]['samples']


        latent_out = self.sample_flux(
            seed, model, positive_conditioning, guidance, sampling_algorithm, scheduling_algorithm,
            steps, denoise, latent_img, max_shift, base_shift, width, height
        )

        print(
            f'\nFlowState Flux Sampler - Sampling Complete.'
            f'\n  - Decoding Batch: {latent_out.shape}\n'
        )

        img_out = vae.decode(latent_out)

        return img_out, latent_out, params

    def execute(self, model, clip_1, clip_2, vae, seed, sampling_algorithm, scheduling_algorithm, guidance, steps, denoise, prompt):

        print(
            f'\n\n\n🌊 FlowState Flux Sampler'
            f'\n  - Sampling...'
        )

        sampling_start_time = time.time()

        image_batch = self.sample(
            model, clip_1, clip_2, vae, seed, sampling_algorithm, scheduling_algorithm, guidance, steps, denoise, prompt
        )

        sampling_duration, sampling_mins, sampling_secs = get_mins_and_secs(sampling_start_time)

        print(
            f'\nFlowState Flux Sampler - Decoding complete.'
            f'\n  - Total Generated Images: {image_batch.shape[0]}'
            f'\n  - Output Resolution: {image_batch.shape[2]} x {image_batch.shape[1]}'
            f'\n  - Generation Time: {sampling_mins}m {sampling_secs}s ({sampling_duration})\n'
        )

        print(
            f'\n  - Sampling copmplete.'
        )

        return (image_batch, )


