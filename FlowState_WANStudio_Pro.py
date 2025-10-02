# Project: FlowState WAN Studio
# Description: All-in-one WAN Video with enhanced options.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'\t - 🟢 🎬 Loaded FlowState WAN Studio Pro.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FlowState_Node import FlowState_Node


##
# OUTSIDE IMPORTS
##
import time, types

from nodes import UNETLoader
from nodes import CLIPLoader
from nodes import VAELoader
from nodes import CLIPTextEncode
from nodes import LoraLoaderModelOnly
from nodes import KSamplerAdvanced
from nodes import VAEDecodeTiled

from comfy_extras.nodes_wan import WanImageToVideo
from comfy_extras.nodes_video import CreateVideo
from comfy_extras.nodes_video import SaveVideo


##
# NODES
##
class FlowState_WANStudio_Pro(FlowState_Node):
    CATEGORY = 'FlowState Creator Suite/Video'
    DESCRIPTION = 'All-in-one WAN Video with enhanced options.'
    FUNCTION = 'execute'
    RETURN_TYPES = TYPE_WAN_STUDIO_OUT
    RETURN_NAMES = ('video', )
    OUTPUT_TOOLTIPS = ('The created video.', )

    def __init__(self):
        super().__init__('🌊🎬 FlowState WAN Studio Pro')

        self.working_high_noise_model = None
        self.working_low_noise_model = None
        self.working_clip = None
        self.working_vae = None

        self.batch_size = 1
        self.latent_batch_in = None
        self.stage_1_latent_batch_out = None
        self.stage_2_latent_batch_out = None
        self.frames_batch_out = None

        self.created_video = None

        self.sampling_params = None

        self.stage_params = {
            'high': {
                'add_noise': 'enable',
                'cfg': 1.0,
                'start_at_step': 0,
                'end_at_step': 2,
                'return_leftover_noise': 'enable',
                'denoise': 1.0,
            },
            'low': {
                'add_noise': 'disable',
                'cfg': 1.0,
                'start_at_step': 2,
                'end_at_step': 4,
                'return_leftover_noise': 'disable',
                'denoise': 1.0,
            }
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                # MODEL SETTINGS
                'model_label': TYPE_FLOWSTATE_LABEL_MODEL,
                'high_noise_model_name': TYPE_DIFFUSION_MODELS_LIST(),
                'low_noise_model_name': TYPE_DIFFUSION_MODELS_LIST(),
                'weight_dtype': TYPE_WEIGHT_DTYPE,
                # MODEL AUGMENTATION SETTINGS
                'aumentation_label': TYPE_FLOWSTATE_LABEL_AUG,
                'sage_attention': TYPE_SAGE_ATTENTION_MODE,
                'high_noise_lora': TYPE_LORAS_LIST(),
                'low_noise_lora': TYPE_LORAS_LIST(),
                'style_lora': TYPE_LORAS_LIST(),
                # ENCODER SETTINGS
                'encoders_label': TYPE_FLOWSTATE_LABEL_ENCODER,
                'clip_name': TYPE_CLIPS_LIST(),
                'vae_name': TYPE_VAES_LIST(),
                # VIDEO SETTINGS
                'video_label': TYPE_FLOWSTATE_LABEL_VIDEO,
                'resolution': TYPE_WAN_STUDIO_RESOLUTION,
                'orientation': TYPE_LATENT_SOURCE_ORIENTATION,
                'custom_width': TYPE_IMG_WIDTH,
                'custom_height': TYPE_IMG_HEIGHT,
                'num_video_frames': TYPE_NUM_VIDEO_FRAMES,
                'fps': TYPE_FPS,
                'format': TYPE_VIDEO_FORMAT,
                'codec': TYPE_VIDEO_CODEC,
                'save_video': TYPE_BOOLEAN_SAVE_VIDEO,
                'filename_prefix': TYPE_WAN_STUDIO_FILENAME_PREFIX,
                # SAMPLING PARAMETERS
                'sampling_label': TYPE_FLOWSTATE_LABEL_SAMPLING,
                'seed': TYPE_SEED,
                'sampling_algorithm': TYPE_SAMPLERS(),
                'scheduling_algorithm': TYPE_SCHEDULERS(),
                'steps': TYPE_STEPS,
                'tiled_decode': TYPE_TILED_DECODE,
                # PROMPT
                'prompt_label': TYPE_FLOWSTATE_LABEL_PROMPT,
                'pos_prompt': TYPE_PROMPT_WAN_STUDIO_POSITIVE,
                'neg_prompt': TYPE_PROMPT_WAN_STUDIO_NEGATIVE,
            },
            'optional': {
                'starting_frame': TYPE_WAN_STUDIO_STARTING_FRAME,
                'audio': TYPE_AUDIO_IN,
                'clip_vision': TYPE_WAN_CLIP_VISION,
            }
        }

    # INITIALIZATION METHODS
    def set_stage_parameters(self):
        total_steps = self.sampling_params['steps']
        middle_step = total_steps // 2
        self.stage_params['high']['end_at_step'] = middle_step
        self.stage_params['low']['start_at_step'] = middle_step
        self.stage_params['low']['end_at_step'] = total_steps

    def set_video_parameters(self):
        horizontal_vid = self.sampling_params['orientation'] == 'Horizontal'
        using_custom = self.sampling_params['resolution'] == 'Custom'
        using_starting_frame = self.sampling_params['resolution'] == 'Use Starting Frame Resolution'
        have_starting_frame = self.sampling_params['starting_frame'] != None
        using_preselected = not using_custom and not using_starting_frame

        self.width_to_use = self.sampling_params['custom_width']
        self.height_to_use = self.sampling_params['custom_height']

        if using_preselected:
            res_split = self.sampling_params['resolution'].split(' - ')[0].split('x')
            self.width_to_use = int(res_split[0] if horizontal_vid else res_split[1])
            self.height_to_use = int(res_split[1] if horizontal_vid else res_split[0])
        
        if using_starting_frame and have_starting_frame:
            starting_frame_dims = self.sampling_params['starting_frame'].shape
            self.width_to_use = starting_frame_dims[2]
            self.height_to_use = starting_frame_dims[1]

        starting_frame_msg = self.sampling_params['starting_frame'].shape if have_starting_frame else 'None'

        self.print_status([
            ('Setting video parameters...',),
            ('Starting Frame', starting_frame_msg),
            ('Video width', self.width_to_use),
            ('Video height', self.height_to_use),
            ('Video frames', self.sampling_params["num_video_frames"]),
            ('Batch size', self.batch_size)
        ])

    # ENCODING METHODS
    def handle_text_encoding(self):
        self.print_status([('Encoding text prompts.',)])
        self.pos_conditioning = CLIPTextEncode().encode(self.working_clip, self.sampling_params['pos_prompt'])[0]
        self.neg_conditioning = CLIPTextEncode().encode(self.working_clip, self.sampling_params['neg_prompt'])[0]

    def handle_encoding(self):
        self.handle_text_encoding()
        self.print_status([('Preparing latent batch.',)])

        pos, neg, latent = WanImageToVideo().execute(
            self.pos_conditioning,
            self.neg_conditioning,
            self.working_vae,
            self.width_to_use,
            self.height_to_use,
            self.sampling_params['num_video_frames'],
            self.batch_size,
            self.sampling_params['starting_frame'],
            clip_vision_output=self.sampling_params['clip_vision']
        )

        self.pos_conditioning = pos
        self.neg_conditioning = neg
        self.latent_batch_in = latent

    def handle_decoding(self):
        self.print_status([
            ('Decoding video...',),
            ('Batch dimensions', self.stage_2_latent_batch_out['samples'].shape)
        ])

        decoding_start = time.time()

        if self.sampling_params['tiled_decode'] == True:
            self.print_status([('Using Tiled Decoding...',)])
            self.frames_batch_out = VAEDecodeTiled().decode(
                self.working_vae,
                self.stage_2_latent_batch_out,
                tile_size=512,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8
            )[0]
        else:
            self.frames_batch_out = self.working_vae.decode(self.stage_2_latent_batch_out['samples'])[0]

        decoding_duration, decoding_mins, decoding_secs = get_mins_and_secs(decoding_start)

        self.print_status([
            ('Decoding Time', f'{decoding_mins}m {decoding_secs}s ({decoding_duration})')
        ])

    # LOADING & PATCHING METHODS
    def handle_loading(self):
        high_noise_model = self.sampling_params['high_noise_model_name']
        low_noise_model = self.sampling_params['low_noise_model_name']
        weight_dtype = self.sampling_params['weight_dtype']

        self.print_status([
            ('Loading high noise model', high_noise_model),
            ('Loading low noise model', low_noise_model),
            ('Loading CLIP', self.sampling_params['clip_name']),
            ('Loading VAE', self.sampling_params['vae_name'])
        ])

        self.working_high_noise_model = UNETLoader().load_unet(high_noise_model, weight_dtype)[0]
        self.working_low_noise_model = UNETLoader().load_unet(low_noise_model, weight_dtype)[0]
        self.working_clip = CLIPLoader().load_clip(self.sampling_params['clip_name'], 'wan', 'default')[0]
        self.working_vae = VAELoader().load_vae(self.sampling_params['vae_name'])[0]

    def patch_sage(self):
        sage_mode = self.sampling_params['sage_attention']
        self.print_status([('Patching models with Sage Attention', sage_mode)])

        self.working_high_noise_model = SageAttention.patch(self.working_high_noise_model, sage_mode)[0]
        self.working_low_noise_model = SageAttention.patch(self.working_low_noise_model, sage_mode)[0]

    def patch_lora(self, stage='both'):
        if stage == 'both':
            self.print_status([('Patching both models with style LoRA', self.sampling_params['style_lora'])])
            
            self.working_high_noise_model = LoraLoaderModelOnly().load_lora_model_only(
                self.working_high_noise_model, self.sampling_params['style_lora'], 1.0
            )[0]

            self.working_low_noise_model = LoraLoaderModelOnly().load_lora_model_only(
                self.working_low_noise_model, self.sampling_params['style_lora'], 1.0
            )[0]
            
            return

        if stage == 'high':
            self.print_status([('Patching High-Noise LoRA', self.sampling_params['high_noise_lora'])])
            self.working_high_noise_model = LoraLoaderModelOnly().load_lora_model_only(
                self.working_high_noise_model, self.sampling_params['high_noise_lora'], 1.0
            )[0]
        else:
            self.print_status([('Patching Low-Noise LoRA', self.sampling_params['low_noise_lora'])])
            self.working_low_noise_model = LoraLoaderModelOnly().load_lora_model_only(
                self.working_low_noise_model, self.sampling_params['low_noise_lora'], 1.0
            )[0]

    def handle_patching(self):
        need_sage = self.sampling_params['sage_attention'] != 'disabled'
        need_high_noise_lora = self.sampling_params['high_noise_lora'] != 'disabled'
        need_low_noise_lora = self.sampling_params['low_noise_lora'] != 'disabled'
        need_style_lora = self.sampling_params['style_lora'] != 'disabled'

        self.print_status([
            ('Checking model patch state...',),
            ('Sage Status', self.sampling_params['sage_attention']),
            ('High-Noise Optimization LoRA Status', self.sampling_params['high_noise_lora']),
            ('Low-Noise Optimization LoRA Status', self.sampling_params['low_noise_lora']),
            ('Style LoRA Status', self.sampling_params['style_lora']),
        ])

        if need_high_noise_lora: self.patch_lora('high')
        if need_low_noise_lora: self.patch_lora('low')
        if need_style_lora: self.patch_lora('both')
        if need_sage: self.patch_sage()

    # SAMPLING METHODS
    def sample(self, stage):
        self.print_status([('Sampling Stage', stage)])

        sampling_start = time.time()

        stages = {
            'high': (self.working_high_noise_model, self.latent_batch_in),
            'low': (self.working_low_noise_model, self.stage_1_latent_batch_out)
        }

        latent_batch_out = KSamplerAdvanced().sample(
            stages[stage][0],
            self.stage_params[stage]['add_noise'],
            self.sampling_params['seed'],
            self.sampling_params['steps'],
            self.stage_params[stage]['cfg'],
            self.sampling_params['sampling_algorithm'],
            self.sampling_params['scheduling_algorithm'],
            self.pos_conditioning,
            self.neg_conditioning,
            stages[stage][1],
            self.stage_params[stage]['start_at_step'],
            self.stage_params[stage]['end_at_step'],
            self.stage_params[stage]['return_leftover_noise'],
            self.stage_params[stage]['denoise']
        )[0]

        if stage == 'high':
            self.stage_1_latent_batch_out = latent_batch_out
        else:
            self.stage_2_latent_batch_out = latent_batch_out

        sampling_duration, sampling_mins, sampling_secs = get_mins_and_secs(sampling_start)

        self.print_status([
            (f'Sampling stage ({stage}) complete.',),
            ('Sampling Stage Time', f'{sampling_mins}m {sampling_secs}s ({sampling_duration})')
        ])

    # SAVE VIDEO
    def create_video(self):
        self.created_video = CreateVideo.execute(
            self.frames_batch_out,
            self.sampling_params['fps'],
            self.sampling_params['audio']
        )[0]
    
    def save_video(self):
        # hidden_inputs = types.SimpleNamespace()
        # hidden_inputs.prompt = self.sampling_params['prompt']
        # hidden_inputs.extra_pnginfo = self.sampling_params['extra_pnginfo']
        # SaveVideo.hidden = hidden_inputs

        # self.saved_video_output = SaveVideo.execute(
        SaveVideo.execute(
            self.created_video,
            self.sampling_params['filename_prefix'],
            self.sampling_params['format'],
            self.sampling_params['codec']
        )

    # MAIN
    def execute(self,
            model_label, high_noise_model_name, low_noise_model_name, weight_dtype,

            aumentation_label, sage_attention, high_noise_lora, low_noise_lora, style_lora,

            encoders_label, clip_name, vae_name,

            video_label, resolution, orientation, custom_width, custom_height, num_video_frames,
            fps, format, codec, save_video, filename_prefix,

            sampling_label, seed, sampling_algorithm, scheduling_algorithm, steps, tiled_decode,

            prompt_label, pos_prompt, neg_prompt,

            starting_frame=None, audio=None, clip_vision=None
        ):

        # PRINT SYSTEM STATUS
        self.print_status([('Preparing sampler...',)], init=True)

        # INITIALIZATION
        self.sampling_params = locals()
        self.set_stage_parameters()
        self.set_video_parameters()

        # SAMPLING START TIME
        sampling_start = time.time()

        # SAMPLING PIPELINE
        self.handle_loading()
        self.handle_encoding()
        self.handle_patching()
        self.sample('high')
        self.sample('low')
        self.handle_decoding()
        self.create_video()

        # OPTIONAL SAVE
        if save_video:
            self.save_video()
     
        # SAMPLING END
        sampling_duration, sampling_mins, sampling_secs = get_mins_and_secs(sampling_start)

        # PRINT SYSTEM STATUS
        self.print_status([
            ('Video generation complete.',),
            ('Video Frames', self.frames_batch_out.shape[0]),
            ('Output Resolution', f'{self.frames_batch_out.shape[2]} x {self.frames_batch_out.shape[1]}'),
            ('Generation Time', f'{sampling_mins}m {sampling_secs}s ({sampling_duration}s)')
        ], end=True)

        return (self.created_video, )

