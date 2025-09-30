# Project: FlowState Painter
# Description: A 2-stage pipeline that uses Qwen Image Edit to alter images, and then Flux to refine for quality enhancement.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'\t - 🟢 👩🏻‍🍳 Loaded Chef.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FlowState_Node import FlowState_Node
from .FlowState_SimpleLatent import *


##
# OUTSIDE IMPORTS
##
import time, copy, math

from nodes import UNETLoader
from nodes import CheckpointLoaderSimple
from nodes import DualCLIPLoader
from nodes import CLIPLoader
from nodes import VAELoader
from nodes import CLIPTextEncode
from nodes import LoraLoaderModelOnly
from nodes import VAEDecodeTiled

from comfy_extras.nodes_custom_sampler import RandomNoise
from comfy_extras.nodes_custom_sampler import BasicGuider
from comfy_extras.nodes_custom_sampler import KSamplerSelect
from comfy_extras.nodes_custom_sampler import BasicScheduler
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_flux import FluxGuidance

from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
from comfy_extras.nodes_cfg import CFGNorm
from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit


##
# NODES
##
class FlowState_Chef(FlowState_Node):
    CATEGORY = 'FlowState Creator Suite/Alteration'
    DESCRIPTION = 'A 2-stage pipeline that uses Qwen Image Edit to alter images, and then Flux to refine for quality enhancement.'
    FUNCTION = 'execute'
    RETURN_TYPES = ('IMAGE', 'LATENT')
    RETURN_NAMES = ('image', 'latent')
    OUTPUT_TOOLTIPS = ('The image output.', 'The latent output.')

    def __init__(self):
        super().__init__('🌊👩🏻‍🍳 FlowState Chef')
        self.qwen_model = None
        self.qwen_clip = None
        self.qwen_vae = None
        self.qwen_style_lora = None
        self.qwen_optimization_lora = None

        self.flux_model = None
        self.flux_clip = None
        self.flux_vae = None
        self.flux_style_lora = None

        self.prev_params = None
        self.sampling_params = None

        self.first_run = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                # PAINTER INGREDIENTS
                'chef_ingredients': TYPE_ANY,

                # MODEL SETTINGS
                'model_label': TYPE_FLOWSTATE_LABEL_MODEL,
                'qwen_model': TYPE_ALL_MODEL_LISTS(),
                'flux_model_filetype': TYPE_MODEL_FILE_TYPE,
                'flux_model': TYPE_ALL_MODEL_LISTS(),
                'weight_dtype': TYPE_WEIGHT_DTYPE,

                # MODEL AUGMENTATION SETTINGS
                'aumentation_label': TYPE_FLOWSTATE_LABEL_AUG,
                'qwen_optimization_lora': TYPE_LORAS_LIST(),
                'qwen_style_lora': TYPE_LORAS_LIST(),
                'qwen_style_lora_strength': TYPE_LORA_STRENGTH,
                'flux_style_lora': TYPE_LORAS_LIST(),
                'flux_style_lora_strength': TYPE_LORA_STRENGTH,
                'sage_attention': TYPE_SAGE_ATTENTION_MODE,

                # ENCODER SETTINGS
                'encoders_label': TYPE_FLOWSTATE_LABEL_ENCODER,
                'qwen_clip': TYPE_CLIPS_LIST(),
                'qwen_vae': TYPE_VAES_LIST(),
                'flux_clip_1': TYPE_CLIPS_LIST(),
                'flux_clip_2': TYPE_CLIPS_LIST(),
                'flux_vae': TYPE_VAES_LIST(),
                'tiled_decode': TYPE_TILED_DECODE,

                # IMAGE SETTINGS
                'image_label': TYPE_FLOWSTATE_LABEL_IMAGE,
                'resolution': TYPE_LATENT_SOURCE_RESOLUTION,
                'orientation': TYPE_LATENT_SOURCE_ORIENTATION,
                'latent_type': TYPE_SIMPLE_LATENT_INPUT_TYPE,
                'custom_width': TYPE_IMG_WIDTH,
                'custom_height': TYPE_IMG_HEIGHT,
                'custom_batch_size': TYPE_LATENT_BATCH_SIZE,

                # GLOBAL SAMPLING PARAMETERS
                'sampling_label': TYPE_FLOWSTATE_LABEL_SAMPLING,
                'seed': TYPE_SEED,
                'sampling_algorithm': TYPE_SAMPLERS(),
                'scheduling_algorithm': TYPE_SCHEDULERS(),

                # QWEN SAMPLING PARAMETERS
                'qwen_steps': TYPE_STEPS,
                'qwen_denoise': TYPE_DENOISE,

                # FLUX SAMPLING PARAMETERS
                'flux_guidance': TYPE_GUIDANCE,
                'flux_steps': TYPE_STEPS,
                'flux_denoise': TYPE_REFINE_DENOISE
            }
        }

    # BATCH/SAMPLER PREP METHODS
    def prepare_batch(self):
        self.print_status([('Preparing latent batch...',)])

        self.latent_batch_in = FlowState_SimpleLatent().execute(
            self.sampling_params['resolution'],
            self.sampling_params['orientation'],
            self.sampling_params['latent_type'],
            self.sampling_params['custom_width'],
            self.sampling_params['custom_height'],
            self.sampling_params['custom_batch_size'],
            self.qwen_vae,
            self.sampling_params['chef_ingredients']['img']
        )[0]
    
    def prepare_sampler_inputs(self):
        self.random_noise = RandomNoise().get_noise(self.sampling_params['seed'])[0]
        self.sampler = KSamplerSelect().get_sampler(self.sampling_params['sampling_algorithm'])[0]
        self.sampling_params['flux_clip'] = f"{self.sampling_params['flux_clip_1']} & {self.sampling_params['flux_clip_2']}"

    def check_stage_params(self, stage='qwen'):
        stage_name = stage.capitalize()
        self.print_status([
            (f'Checking {stage_name} Stage Params...',)
        ])

        check_params = {
            'qwen': [
                'qwen_model', 'qwen_optimization_lora', 'qwen_style_lora', 'qwen_style_lora_strength',
                'qwen_clip', 'qwen_vae', 'qwen_steps', 'qwen_denoise'
            ],
            'flux': [
                'flux_model', 'flux_model_filetype', 'flux_style_lora', 'flux_style_lora_strength',
                'flux_clip', 'flux_vae', 'flux_guidance', 'flux_steps', 'flux_denoise'
            ],
            'shared': [
                'weight_dtype', 'sage_attention', 'tiled_decode', 'resolution', 'orientation',
                'latent_type', 'custom_width', 'custom_height', 'custom_batch_size', 'seed',
                'sampling_algorithm', 'scheduling_algorithm'
            ]
        }

        full_params = check_params[stage] + check_params['shared']
        changed_params = []

        run_stage = False

        if self.first_run:
            run_stage = True
            changed_params.append('FIRST_RUN')
        else:
            qwen_prompt = self.sampling_params['chef_ingredients']['qwen']
            flux_prompt = self.sampling_params['chef_ingredients']['flux']

            prev_qwen_prompt = self.prev_params['chef_ingredients']['qwen']
            prev_flux_prompt = self.prev_params['chef_ingredients']['flux']

            flux_prompt_changed = flux_prompt != prev_flux_prompt
            qwen_prompt_changed = qwen_prompt != prev_qwen_prompt

            is_flux_stage = stage == 'flux'
            is_flux_stage_and_flux_prompt_changed = is_flux_stage and flux_prompt_changed

            if qwen_prompt_changed or is_flux_stage_and_flux_prompt_changed:
                run_stage = True
                changed_params.append('prompt')

            input_img = self.sampling_params['chef_ingredients']['img']
            prev_img = self.prev_params['chef_ingredients']['img']
            
            input_img_shape = input_img.shape
            prev_img_shape = prev_img.shape
            same_size = input_img_shape == prev_img_shape

            if not same_size:
                run_stage = True
                changed_params.append('input_img')
            
            if same_size:
                input_img_changed = not input_img.equal(prev_img)

                if input_img_changed:
                    run_stage = True
                    changed_params.append('input_img')

            for key in full_params:
                param_changed = self.prev_params[key] != self.sampling_params[key]
                if param_changed:
                    changed_params.append(key)
                    run_stage = True

        self.print_status([
            (f'Running {stage_name} Stage', run_stage),
            (f'Changed Params', changed_params),
        ])

        return run_stage

    # QWEN PREP METHODS
    def encode_qwen_prompt(self):
        self.qwen_conditioning = TextEncodeQwenImageEdit().encode(
                self.qwen_clip,
                self.sampling_params['chef_ingredients']['qwen'],
                self.qwen_vae,
                self.sampling_params['chef_ingredients']['img']
            )[0]

    def load_qwen_model(self):
        self.print_status([
            ('Loading Qwen model', self.sampling_params['qwen_model']),
        ])

        self.qwen_model = UNETLoader().load_unet(
            self.sampling_params['qwen_model'], self.sampling_params['weight_dtype']
        )[0]

    def unload_qwen_model(self):
        self.print_status([('Unloading Qwen model', )])
        self.qwen_model = None

    def unload_qwen_clip(self):
        self.print_status([('Unloading Qwen CLIP', )])
        self.qwen_clip = None
    
    def unload_qwen_vae(self):
        self.print_status([('Unloading Qwen VAE', )])
        self.qwen_vae = None

    def unload_qwen_all(self):
        self.print_status([('Unloading all Qwen models', )])
        self.unload_qwen_model()
        self.unload_qwen_clip()
        self.unload_qwen_vae()

    def load_qwen_clip(self):
        self.print_status([
            ('Loading Qwen CLIP', self.sampling_params['qwen_clip']),
        ])

        self.qwen_clip = CLIPLoader().load_clip(
            self.sampling_params['qwen_clip'], 'qwen_image', 'default'
        )[0]

    def load_qwen_vae(self):
        self.print_status([
            ('Loading Qwen VAE', self.sampling_params['qwen_vae']),
        ])

        self.qwen_vae = VAELoader().load_vae(self.sampling_params['qwen_vae'])[0]

    def patch_qwen(self):
        patching_sage = self.sampling_params['sage_attention'] != 'disabled'
        patching_style_lora = self.sampling_params['qwen_style_lora'] != 'disabled'
        patching_optimization_lora = self.sampling_params['qwen_optimization_lora'] != 'disabled'

        if patching_sage:
            self.print_status([
                ('Patching Qwen for Sage Attention', self.sampling_params['sage_attention']),
            ])
            self.qwen_model = SageAttention.patch(
                self.qwen_model, self.sampling_params['sage_attention']
            )[0]
        
        if patching_style_lora:
            self.print_status([
                ('Patching Qwen style LoRA', self.sampling_params['qwen_style_lora']),
            ])
            self.qwen_model = LoraLoaderModelOnly().load_lora_model_only(
                self.qwen_model,
                self.sampling_params['qwen_style_lora'],
                self.sampling_params['qwen_style_lora_strength']
            )[0]

        if patching_optimization_lora:
            self.print_status([
                ('Patching Qwen optimization LoRA', self.sampling_params['qwen_optimization_lora']),
            ])
            self.qwen_model = LoraLoaderModelOnly().load_lora_model_only(
                self.qwen_model,
                self.sampling_params['qwen_optimization_lora'],
                strength_model=1.0
            )[0]

    def check_qwen(self):
        self.print_status([
            (f'Checking Qwen Stage Models...',)
        ])

        model_changed = self.prev_params['qwen_model'] != self.sampling_params['qwen_model']
        clip_changed = self.prev_params['qwen_clip'] != self.sampling_params['qwen_clip']
        vae_changed = self.prev_params['qwen_vae'] != self.sampling_params['qwen_vae']

        style_lora_changed = self.prev_params['qwen_style_lora'] != self.sampling_params['qwen_style_lora']
        optimization_lora_changed = self.prev_params['qwen_optimization_lora'] != self.sampling_params['qwen_optimization_lora']
        sage_changed = self.prev_params['sage_attention'] != self.sampling_params['sage_attention']

        patch_changed = style_lora_changed or optimization_lora_changed or sage_changed

        self.print_status([
            ('Previously Loaded Qwen Models & Patches', ),
            ('Qwen Model', self.prev_params['qwen_model']),
            ('Qwen CLIP', self.prev_params['qwen_clip']),
            ('Qwen VAE', self.prev_params['qwen_vae']),
            ('Qwen Style LoRA', self.prev_params['qwen_style_lora']),
            ('Qwen Optimization LoRA', self.prev_params['qwen_optimization_lora']),
            ('Qwen Sage Attention', self.prev_params['sage_attention']),
        ])

        self.print_status([
            ('Incoming Qwen Models & Patches', ),
            ('Qwen Model', self.sampling_params['qwen_model']),
            ('Qwen CLIP', self.sampling_params['qwen_clip']),
            ('Qwen VAE', self.sampling_params['qwen_vae']),
            ('Qwen Style LoRA', self.sampling_params['qwen_style_lora']),
            ('Qwen Optimization LoRA', self.sampling_params['qwen_optimization_lora']),
            ('Qwen Sage Attention', self.sampling_params['sage_attention']),
        ])

        self.print_status([
            ('Changed Qwen Models & Patches', ),
            ('Qwen Model', model_changed),
            ('Qwen CLIP', clip_changed),
            ('Qwen VAE', vae_changed),
            ('Qwen Style LoRA', style_lora_changed),
            ('Qwen Optimization LoRA', optimization_lora_changed),
            ('Qwen Sage Attention', sage_changed),
        ])

        return model_changed, patch_changed, clip_changed, vae_changed

    def prepare_qwen(self):
        model_changed = True
        patch_changed = True
        clip_changed = True
        vae_changed = True

        if self.first_run:
            self.print_status([
                ('First run.', ),
                ('Loading all Qwen models...', )
            ])
        else:
            model_changed, patch_changed, clip_changed, vae_changed = self.check_qwen()

        if model_changed or patch_changed:
            if not self.first_run: self.unload_qwen_model()
            self.load_qwen_model()
            self.patch_qwen()
        
        if clip_changed:
            if not self.first_run: self.unload_qwen_clip()
            self.load_qwen_clip()

        if vae_changed:
            if not self.first_run: self.unload_qwen_vae()
            self.load_qwen_vae()

    # FLUX PREP METHODS
    def encode_flux_prompt(self):
        self.flux_conditioning = CLIPTextEncode().encode(
            self.flux_clip,
            self.sampling_params['chef_ingredients']['flux']
        )[0]

    def load_flux_model(self):
        self.print_status([
            ('Loading Flux model', self.sampling_params['flux_model'])
        ])

        try:
            self.flux_model = UNETLoader().load_unet(
                self.sampling_params['flux_model'],
                self.sampling_params['weight_dtype']
            )[0]
        except:
            raise ValueError(self.format_value_error('checkpoint'))

    def load_flux_clip(self):
        self.print_status([
            ('Loading Flux CLIPs', self.sampling_params['flux_clip'])
        ])

        self.flux_clip = DualCLIPLoader().load_clip(
            self.sampling_params['flux_clip_1'],
            self.sampling_params['flux_clip_2'],
            'flux', 'default'
        )[0]

    def load_flux_vae(self):
        self.print_status([
            ('Loading Flux VAE', self.sampling_params['flux_vae'])
        ])

        self.flux_vae = VAELoader().load_vae(self.sampling_params['flux_vae'])[0]

    def load_flux_checkpoint(self):
        self.print_status([
            ('Loading Flux Checkpoint', self.sampling_params['flux_model'])
        ])

        try:
            self.flux_checkpoint = CheckpointLoaderSimple().load_checkpoint(self.sampling_params['flux_model'])
        except:
            raise ValueError(self.format_value_error('checkpoint'))

        self.flux_model = self.flux_checkpoint[0]
        self.flux_clip = self.flux_checkpoint[1]
        self.flux_vae = self.flux_checkpoint[2]

    def unload_flux_model(self):
        self.print_status([('Unloading Flux model', )])
        self.flux_model = None

    def unload_flux_clip(self):
        self.print_status([('Unloading Flux CLIP', )])
        self.flux_clip = None
    
    def unload_flux_vae(self):
        self.print_status([('Unloading Flux VAE', )])
        self.flux_vae = None

    def unload_flux_all(self):
        self.print_status([('Unloading all Flux models', )])
        self.unload_flux_model()
        self.unload_flux_clip()
        self.unload_flux_vae()

    def patch_flux(self):
        patching_sage = self.sampling_params['sage_attention'] != 'disabled'
        patching_style_lora = self.sampling_params['flux_style_lora'] != 'disabled'

        if patching_sage:
            self.print_status([
                ('Patching Flux for Sage Attention', self.sampling_params['sage_attention']),
            ])
            self.flux_model = SageAttention.patch(
                self.flux_model, self.sampling_params['sage_attention']
            )[0]
        
        if patching_style_lora:
            self.print_status([
                ('Patching Flux style LoRA', self.sampling_params['flux_style_lora']),
            ])
            self.flux_model = LoraLoaderModelOnly().load_lora_model_only(
                self.flux_model,
                self.sampling_params['flux_style_lora'],
                self.sampling_params['flux_style_lora_strength']
            )[0]

    def check_flux(self):
        self.print_status([
            (f'Checking Flux Stage Models...',)
        ])

        model_changed = self.prev_params['flux_model'] != self.sampling_params['flux_model']
        clip_changed = self.prev_params['flux_clip'] != self.sampling_params['flux_clip']
        vae_changed = self.prev_params['flux_vae'] != self.sampling_params['flux_vae']

        style_lora_changed = self.prev_params['flux_style_lora'] != self.sampling_params['flux_style_lora']
        sage_changed = self.prev_params['sage_attention'] != self.sampling_params['sage_attention']

        patch_changed = style_lora_changed or sage_changed

        self.print_status([
            ('Previously Loaded Flux Models & Patches', ),
            ('Flux Model', self.prev_params['flux_model']),
            ('Flux CLIP', self.prev_params['flux_clip']),
            ('Flux VAE', self.prev_params['flux_vae']),
            ('Flux Style LoRA', self.prev_params['flux_style_lora']),
            ('Flux Sage Attention', self.prev_params['sage_attention']),
        ])

        self.print_status([
            ('Incoming Flux Models & Patches', ),
            ('Flux Model', self.sampling_params['flux_model']),
            ('Flux CLIP', self.sampling_params['flux_clip']),
            ('Flux VAE', self.sampling_params['flux_vae']),
            ('Flux Style LoRA', self.sampling_params['flux_style_lora']),
            ('Flux Sage Attention', self.sampling_params['sage_attention']),
        ])

        self.print_status([
            ('Changed Flux Models & Patches', ),
            ('Flux Model', model_changed),
            ('Flux CLIP', clip_changed),
            ('Flux VAE', vae_changed),
            ('Flux Style LoRA', style_lora_changed),
            ('Flux Sage Attention', sage_changed),
        ])

        return model_changed, patch_changed, clip_changed, vae_changed

    def prepare_flux(self):
        is_checkpoint = self.sampling_params['flux_model_filetype'] == 'checkpoint'

        model_changed = True
        patch_changed = True
        clip_changed = True
        vae_changed = True

        if self.first_run:
            self.print_status([
                ('First run.', ),
                ('Loading all Flux models...', )
            ])
        else:
            model_changed, patch_changed, clip_changed, vae_changed = self.check_flux()

        if model_changed or patch_changed:
            if not self.first_run: self.unload_flux_model()
            if is_checkpoint:
                self.load_flux_checkpoint()
            else:
                self.load_flux_model()
            
            self.patch_flux()

        if clip_changed and not is_checkpoint:
            if not self.first_run: self.unload_flux_clip()
            self.load_flux_clip()

        if vae_changed and not is_checkpoint:
            if not self.first_run: self.unload_flux_vae()
            self.load_flux_vae()

    # SAMPLING METHODS
    def sample_qwen(self):
        self.print_status([('Sampling Qwen...',)])

        self.encode_qwen_prompt()

        guider = BasicGuider().get_guider(self.qwen_model, self.qwen_conditioning)[0]

        sigmas = BasicScheduler().get_sigmas(
            self.qwen_model,
            self.sampling_params['scheduling_algorithm'],
            self.sampling_params['qwen_steps'],
            self.sampling_params['qwen_denoise']
        )[0]
        
        self.qwen_latent_batch_out = SamplerCustomAdvanced().sample(
            self.random_noise, guider, self.sampler, sigmas, self.latent_batch_in
        )[1]

        self.print_status([
            ('Qwen Sampling Complete.',),
            ('Decoding Batch', self.qwen_latent_batch_out['samples'].shape)
        ])

        if self.sampling_params['tiled_decode'] == True:
            self.qwen_img_batch_out = VAEDecodeTiled().decode(
                self.qwen_vae,
                self.qwen_latent_batch_out,
                tile_size=512,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8
            )[0]
        else:
            self.qwen_img_batch_out = self.qwen_vae.decode(self.qwen_latent_batch_out['samples'])[0]

    def sample_flux(self):
        self.print_status([('Sampling Flux...',)])

        self.encode_flux_prompt()

        guided_conditioning = FluxGuidance().append(
            self.flux_conditioning, 
            self.sampling_params['flux_guidance']
        )[0]
        
        guider = BasicGuider().get_guider(self.flux_model, guided_conditioning)[0]

        sigmas = BasicScheduler().get_sigmas(
            self.flux_model,
            self.sampling_params['scheduling_algorithm'],
            self.sampling_params['flux_steps'],
            self.sampling_params['flux_denoise']
        )[0]

        self.flux_latent_batch_in = {'samples': self.flux_vae.encode(self.qwen_img_batch_out)}
        
        self.flux_latent_batch_out = SamplerCustomAdvanced().sample(
            self.random_noise, guider, self.sampler, sigmas, self.flux_latent_batch_in
        )[1]

        self.print_status([
            ('Flux Sampling Complete.',),
            ('Decoding Batch', self.flux_latent_batch_out['samples'].shape)
        ])

        if self.sampling_params['tiled_decode'] == True:
            self.flux_img_batch_out = VAEDecodeTiled().decode(
                self.flux_vae,
                self.flux_latent_batch_out,
                tile_size=512,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8
            )[0]
        else:
            self.flux_img_batch_out = self.flux_vae.decode(self.flux_latent_batch_out)

    def run_stages(self):
        self.prepare_sampler_inputs()
        self.prepare_batch()

        for stage in ['qwen', 'flux']:
            start_time = time.time()

            running_stage = self.check_stage_params(stage)

            if stage == 'qwen' and running_stage:
                self.prepare_qwen()
                self.sample_qwen()
            
            if stage == 'flux' and running_stage:
                self.prepare_flux()
                self.sample_flux()

            duration, mins, secs = self.get_mins_and_secs(start_time)

            stage_name = stage.capitalize() + ' Stage'
            self.print_status([
                (stage_name,),
                ('Generation Time', f'{mins}m {secs}s ({duration}s)')
            ])

        self.prev_params = self.sampling_params
        self.first_run = False

    def execute(self,
            chef_ingredients,

            model_label, qwen_model, flux_model_filetype, flux_model, weight_dtype,

            aumentation_label, qwen_optimization_lora, qwen_style_lora, qwen_style_lora_strength,
            flux_style_lora, flux_style_lora_strength, sage_attention,

            encoders_label, qwen_clip, qwen_vae, flux_clip_1, flux_clip_2, flux_vae, tiled_decode,

            image_label, resolution, orientation, latent_type, custom_width, custom_height, custom_batch_size,

            sampling_label, seed, sampling_algorithm, scheduling_algorithm,
            qwen_steps, qwen_denoise, flux_guidance, flux_steps, flux_denoise
        ):

        self.print_status([('Preparing FlowState Chef...',)], init=True)

        self.sampling_params = locals()

        start_time = time.time()

        self.run_stages()

        duration, mins, secs = self.get_mins_and_secs(start_time)
        
        self.print_status([
            ('Run Time', f'{mins}m {secs}s ({duration}s)')
        ], end=True)

        return (self.flux_img_batch_out, self.flux_latent_batch_out)

