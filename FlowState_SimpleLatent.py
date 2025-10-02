# Project: FlowState Simple Latent
# Description: Select from input image to create a new batch of latent images, or select an empty latent.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng



##
# SYSTEM STATUS
##
print(f'\t - 🟢 👌 Loaded FlowState Simple Latent.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FlowState_Node import FlowState_Node


##
# OUTSIDE IMPORTS
##
import torch
from comfy import model_management


##
# NODES
##
class FlowState_SimpleLatent(FlowState_Node):
    CATEGORY = 'FlowState Creator Suite/Latent'
    DESCRIPTION = 'Create a new batch of latent images to be denoised via sampling.'
    FUNCTION = 'execute'
    RETURN_TYPES = TYPE_LATENT_SOURCE_OUT
    RETURN_NAMES = ('Latent Image',)
    OUTPUT_TOOLTIPS = ('The latent image batch.',)

    def __init__(self):
        super().__init__('🌊👌 FlowState Simple Latent')
        self.device = model_management.intermediate_device()
        self.latent_channels = 4

        self.have_input_image = False
        self.using_empty_latent = False
        self.using_input_image = False
        self.using_horizontal = False
        self.using_custom_resolution = False

        self.empty_latent = None
        self.input_latent = None

        self.width_to_use = None
        self.height_to_use = None

        self.system_message = None
        self.latent_batch_out = None

        self.batch_params = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'resolution': TYPE_LATENT_SOURCE_RESOLUTION,
                'orientation': TYPE_LATENT_SOURCE_ORIENTATION,
                'latent_type': TYPE_SIMPLE_LATENT_INPUT_TYPE,
                'custom_width': TYPE_IMG_WIDTH,
                'custom_height': TYPE_IMG_HEIGHT,
                'custom_batch_size': TYPE_LATENT_BATCH_SIZE,
                'vae': TYPE_VAE_IN
            },
            'optional': {
                'input_img': TYPE_IMAGE
            }
        }

    def generate_empty_latent(self):
        self.empty_latent = torch.zeros([
            self.batch_params['custom_batch_size'],
            self.latent_channels,
            self.height_to_use // 8,
            self.width_to_use // 8
        ], device=self.device)

    def set_img_parameters(self):
        self.using_empty_latent = self.batch_params['latent_type'] == 'Empty Latent'
        self.using_input_image = self.batch_params['latent_type'] == 'Input Image'

        self.have_input_image = self.batch_params['input_img'] != None
        self.using_horizontal = self.batch_params['orientation'] == 'Horizontal'

        self.using_custom_resolution = self.batch_params['resolution'] == 'Custom'
        self.using_resolution_selection = not self.using_custom_resolution

    def set_resolution(self, width, height):
        self.width_to_use = width
        self.height_to_use = height

    def set_batch_out(self, msg, latent):
        self.system_message = msg
        self.latent_batch_out = {'samples': latent}

    def prepare_empty_latent(self):
        if self.using_custom_resolution:
            self.set_resolution(self.batch_params['custom_width'], self.batch_params['custom_height'])

        if self.using_resolution_selection:
            res_split = self.batch_params['resolution'].split(' - ')[0].split('x')
            width = int(res_split[0] if self.using_horizontal else res_split[1])
            height = int(res_split[1] if self.using_horizontal else res_split[0])
            self.set_resolution(width, height)

        self.generate_empty_latent()

    def prepare_latent_batch(self):
        if self.using_empty_latent:
            self.prepare_empty_latent()
            self.set_batch_out('empty latent', self.empty_latent)
        
        if self.using_input_image and not self.have_input_image:
            self.prepare_empty_latent()
            self.set_batch_out('empty latent. No input image', self.empty_latent)
        
        if self.using_input_image and self.have_input_image:
            self.input_latent = self.batch_params['vae'].encode(self.batch_params['input_img'][:,:,:,:3])
            self.set_batch_out('latent from input image', self.input_latent)

    def execute(self, resolution, orientation, latent_type, custom_width, custom_height, custom_batch_size, vae, input_img=None):
        self.print_status([('Preparing latent batch...',)], init=True)

        batch_start_time = time.time()

        self.batch_params = locals()

        self.set_img_parameters()
        self.prepare_latent_batch()

        batch_duration, batch_mins, batch_secs = get_mins_and_secs(batch_start_time)

        self.print_status([
            (f'Prepared {self.system_message}.', self.latent_batch_out['samples'].shape),
            ('Latent Batch Size', self.batch_params['custom_batch_size']),
            ('Preparation Time', f'{batch_mins}m {batch_secs}s ({batch_duration}s)'),
        ], end=True)

        return (self.latent_batch_out,)

