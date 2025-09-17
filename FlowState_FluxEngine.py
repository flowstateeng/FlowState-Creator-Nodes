# Project: FlowState FluxEngine
# Description: All-in-one Flux.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'\t - 🟢 🚒 Loaded Flux Engine.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FlowState_LatentSource import *


##
# OUTSIDE IMPORTS
##
import time, copy, math

from nodes import UNETLoader
from nodes import CheckpointLoaderSimple
from nodes import DualCLIPLoader
from nodes import VAELoader
from nodes import CLIPTextEncode
from nodes import LoraLoaderModelOnly

from comfy_extras.nodes_custom_sampler import RandomNoise
from comfy_extras.nodes_custom_sampler import BasicGuider
from comfy_extras.nodes_custom_sampler import KSamplerSelect
from comfy_extras.nodes_custom_sampler import BasicScheduler
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_flux import FluxGuidance


##
# NODES
##
class FlowState_FluxEngine:
    CATEGORY = 'FlowState Creator Suite/Flux'
    DESCRIPTION = 'Simple sampler for Flux models.'
    FUNCTION = 'execute'
    RETURN_TYPES = TYPE_FLUX_ENGINE_OUT
    RETURN_NAMES = ('model', 'clip', 'vae', 'image', 'latent', )
    OUTPUT_TOOLTIPS = (
        'The selected Diffusion Model.',
        'The selected CLIP.',
        'The selected VAE.',
        'The image batch.',
        'The latent batch.',
    )

    def __init__(self):
        self.working_model = None
        self.working_model_name = None
        self.sage_patched = False
        self.lora_patched = False

        self.working_clip = None
        self.working_clip_name = None

        self.working_vae = None
        self.working_vae_name = None

        self.sampling_params = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'model_filetype': TYPE_MODEL_FILE_TYPE,
                'model_name': TYPE_ALL_MODEL_LISTS(),
                'weight_dtype': TYPE_WEIGHT_DTYPE,
                'sage_attention': TYPE_SAGE_ATTENTION_MODE,
                'lora_model': TYPE_LORAS_LIST(),
                'lora_strength': TYPE_LORA_STRENGTH,
                'clip_1_name': TYPE_CLIPS_LIST(),
                'clip_2_name': TYPE_CLIPS_LIST(),
                'vae_name': TYPE_VAES_LIST(),
                'resolution': TYPE_LATENT_SOURCE_RESOLUTION,
                'orientation': TYPE_LATENT_SOURCE_ORIENTATION,
                'latent_type': TYPE_LATENT_SOURCE_INPUT_TYPE,
                'custom_width': TYPE_IMG_WIDTH,
                'custom_height': TYPE_IMG_HEIGHT,
                'custom_batch_size': TYPE_LATENT_BATCH_SIZE,
                'seed': TYPE_SEED,
                'sampling_algorithm': TYPE_SAMPLERS(),
                'scheduling_algorithm': TYPE_SCHEDULERS(),
                'guidance': TYPE_GUIDANCE,
                'steps': TYPE_STEPS,
                'denoise': TYPE_DENOISE,
                'prompt': TYPE_PROMPT_POSITIVE,
                'image': TYPE_INPUT_FILES(),
                # 'add_params': BOOLEAN_PARAMS,
                # 'add_prompt': BOOLEAN_PROMPT,
                # 'show_params_in_terminal': BOOLEAN_PARAMS_TERM,
                # 'show_prompt_in_terminal': BOOLEAN_PROMPT_TERM,
                # 'font_size': FONT_SIZE,
            },
            'optional': {
                'input_img': TYPE_IMAGE
                # 'added_lines': ADDED_LINES,
                # 'seed_str_list': SEED_LIST,
            }
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
            f'\n 🌊🚒 FlowState Flux Engine - Adding Params.'
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

    def reset_model(self):
        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Unloading model...'
        )
        self.working_model = None
        self.working_model_name = None
        self.sage_patched = False
        self.lora_patched = False

    def reset_clip(self):
        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Unloading CLIP...'
        )
        self.working_clip = None
        self.working_clip_name = None

    def reset_vae(self):
        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Unloading VAE...'
        )
        self.working_vae = None
        self.working_vae_name = None

    def reset_all(self):
        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Unloading model, CLIP & VAE...'
        )
        self.reset_model()
        self.reset_clip()
        self.reset_vae()
        self.sampling_params = None

    def handle_changing(self):
        first_run = self.working_model == None and self.working_clip == None and self.working_vae == None

        if not first_run:
            possible_clip_names = [
                f'{self.sampling_params['clip_1_name']} & {self.sampling_params['clip_2_name']}',
                self.sampling_params['model_name']
            ]

            changing_model = self.working_model_name != self.sampling_params['model_name'] or self.working_model == None
            changing_clip = self.working_clip_name not in possible_clip_names or self.working_clip == None
            changing_vae = self.working_vae_name != self.sampling_params['vae_name'] or self.working_vae == None

            change_model_msg = 'CHANGING_MODEL' if changing_model else 'KEEPING_MODEL'
            change_clip_msg = 'CHANGING_CLIP' if changing_clip else 'KEEPING_CLIP'
            change_vae_msg = 'CHANGING_VAE' if changing_vae else 'KEEPING_VAE'

            print(
                f'\n 🌊🚒 FlowState Flux Engine'
                f'\n  - Checking change state...'
                f'\n  - Changing: {change_model_msg} & {change_clip_msg} & {change_vae_msg}\n'
            )

            if changing_model: self.reset_model()
            if changing_clip: self.reset_clip()
            if changing_vae: self.reset_vae()

    def handle_loading(self):
        self.handle_changing()

        model_state = 'MODEL_LOADED' if self.working_model != None else 'MODEL_UNLOADED'
        clip_state = 'CLIP_LOADED' if self.working_clip != None else 'CLIP_UNLOADED'
        vae_state = 'VAE_LOADED' if self.working_vae != None else 'VAE_UNLOADED'

        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Checking load state...'
            f'\n  - Status: {model_state} & {clip_state} & {vae_state}\n'
        )

        is_checkpoint = self.sampling_params['model_filetype'] == 'checkpoint'
        model_name = self.sampling_params['model_name']
        clip_names = model_name if is_checkpoint else f'{self.sampling_params['clip_1_name']} & {self.sampling_params['clip_2_name']}'
        vae_name = model_name if is_checkpoint else self.sampling_params['vae_name']

        model_is_loaded = self.working_model != None
        model_has_changed = self.working_model_name != model_name

        clip_is_loaded = self.working_clip != None
        clip_has_changed = self.working_clip_name != clip_names

        vae_is_loaded = self.working_vae != None
        vae_has_changed = self.working_vae_name != vae_name

        model_action = f'Pre-loaded model: {model_name}'
        clip_action = f'Pre-loaded CLIP: {clip_names}'
        vae_action = f'Pre-loaded VAE: {vae_name}'

        error_message = (
            f'\n\n{"-" * 100}'
            f'\n (ERROR) 🌊🚒 FlowState Flux Engine'
            f"\n - Error loading {model_name}. Are you sure it's a {self.sampling_params['model_filetype']}?"
            f"\n - Be sure to select the right 'model_filetype' for the model you're selecting."
            f'\n{"-" * 100}\n'
        )

        if not model_is_loaded or model_has_changed:
            self.working_model_name = model_name

            if is_checkpoint:
                print(f'  - Loaded checkpoint: {model_name}...\n')
                try:
                    checkpoint = CheckpointLoaderSimple().load_checkpoint(model_name)
                    self.working_model = checkpoint[0]
                except:
                    self.reset_all(error_message)
                    raise ValueError(error_message)


                self.working_clip = checkpoint[1]
                self.working_clip_name = model_name

                self.working_vae = checkpoint[2]
                self.working_vae_name = model_name

                return 

            try:
                self.working_model = UNETLoader().load_unet(model_name, self.sampling_params['weight_dtype'])[0]
                model_action = f'Loaded model: {model_name}'
            except:
                self.reset_all(error_message)
                raise ValueError(error_message)


        if not clip_is_loaded or clip_has_changed:
            self.working_clip = DualCLIPLoader().load_clip(self.sampling_params['clip_1_name'], self.sampling_params['clip_2_name'], 'flux', 'default')[0]
            self.working_clip_name = f'{clip_names}'
            clip_action = f'Loaded CLIP: {clip_names}'

        if not vae_is_loaded or vae_has_changed:
            self.working_vae = VAELoader().load_vae(self.sampling_params['vae_name'])[0]
            self.working_vae_name = vae_name
            vae_action = f'Loaded VAE: {vae_name}'

        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - {model_action}'
            f'\n  - {clip_action}'
            f'\n  - {vae_action}\n'
        )

    def patch_sage(self):
        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Patching model with Sage Attention: {self.sampling_params["sage_attention"]}\n'
        )
        self.working_model = SageAttention.patch(self.working_model, self.sampling_params['sage_attention'])[0]
        self.sage_patched = True

    def patch_lora(self):
        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Patching model with LoRA: {self.sampling_params["lora_model"]}\n'
        )
        self.working_model = LoraLoaderModelOnly().load_lora_model_only(
            self.working_model, self.sampling_params['lora_model'], self.sampling_params['lora_strength']
        )[0]
        self.lora_patched = True

    def handle_patching(self):
        sage_state = 'sage_active' if self.sage_patched == True else 'sage_disabled'
        lora_state = 'lora_active' if self.lora_patched == True else 'lora_disabled'

        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Checking model patch state...'
            f'\n  - Status: {sage_state} & {lora_state}\n'
        )
        need_sage = self.sampling_params['sage_attention'] != 'disabled' and self.sage_patched == False
        need_lora = self.sampling_params['lora_model'] != 'none' and self.lora_patched == False

        need_sage_but_not_lora = need_sage and not need_lora
        need_lora_but_not_sage = need_lora and not need_sage
        need_both = need_sage and need_lora

        need_to_remove_sage = self.sampling_params['sage_attention'] == 'disabled' and self.sage_patched == True
        need_to_remove_lora = self.sampling_params['lora_model'] == 'none' and self.lora_patched == True

        if need_to_remove_sage or need_to_remove_lora:
            need_to_remove = 'sage & lora' if need_to_remove_lora and need_to_remove_sage else ('sage' if need_to_remove_sage else 'lora')
            print(
                f'  - Need to remove: {need_to_remove}'
                f'\n  - Reloading model...'
            )
            self.working_model = None
            self.sage_patched = False
            self.lora_patched = False
            self.handle_loading()

        if need_sage_but_not_lora:
            self.patch_sage()

        if need_lora_but_not_sage:
            self.patch_lora()

        if need_both:
            self.patch_lora()
            self.patch_sage()

    def sample(self, latent_batch_in):
        print(
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Sampling...\n'
        )   

        random_noise = RandomNoise().get_noise(self.sampling_params['seed'])[0]
        conditioning = CLIPTextEncode().encode(self.working_clip, self.sampling_params['prompt'])[0]
        guided_conditioning = FluxGuidance().append(conditioning, self.sampling_params['guidance'])[0]
        guider = BasicGuider().get_guider(self.working_model, guided_conditioning)[0]
        sampler = KSamplerSelect().get_sampler(self.sampling_params['sampling_algorithm'])[0]
        sigmas = BasicScheduler().get_sigmas(
            self.working_model, self.sampling_params['scheduling_algorithm'], self.sampling_params['steps'], self.sampling_params['denoise']
        )[0]
        
        latent_batch_out = SamplerCustomAdvanced().sample(
            random_noise, guider, sampler, sigmas, latent_batch_in
        )[1]['samples']

        print(
            f'\n 🌊🚒 FlowState Flux Engine - Sampling Complete.'
            f'\n - Decoding Batch: {latent_batch_out.shape}\n'
        )

        img_batch_out = self.working_vae.decode(latent_batch_out)

        return img_batch_out, latent_batch_out

    def execute(
            self, model_filetype, model_name, weight_dtype, sage_attention, lora_model, lora_strength, clip_1_name,
            clip_2_name, vae_name, resolution, orientation, latent_type, custom_width, custom_height, custom_batch_size,
            image, seed, sampling_algorithm, scheduling_algorithm, guidance, steps, denoise, prompt, input_img=None
        ):

        print(
            f'\n\n\n  --- START ---\n'
            f'\n 🌊🚒 FlowState Flux Engine'
            f'\n  - Preparing sampler...'
        )

        self.sampling_params = locals()

        self.handle_loading()
        self.handle_patching()

        latent_batch_in = FlowState_LatentSource().execute(
            resolution, orientation, latent_type, custom_width, custom_height,
            custom_batch_size, image, self.working_vae, input_img
        )[0]

        sampling_start_time = time.time()

        img_batch_out, latent_batch_out = self.sample(latent_batch_in)
        
        sampling_duration, sampling_mins, sampling_secs = get_mins_and_secs(sampling_start_time)

        print(
            f'\n 🌊🚒 FlowState Flux Engine - Decoding complete.'
            f'\n  - Total Generated Images: {img_batch_out.shape[0]}'
            f'\n  - Output Resolution: {img_batch_out.shape[2]} x {img_batch_out.shape[1]}'
            f'\n  - Generation Time: {sampling_mins}m {sampling_secs}s ({sampling_duration})\n'
            f'\n\n --- END --- \n\n\n'
        )

        return (self.working_model, self.working_clip, self.working_vae, img_batch_out, {'samples': latent_batch_out}, )


