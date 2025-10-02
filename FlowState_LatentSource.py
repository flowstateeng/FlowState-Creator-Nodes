# Project: FlowState Latent Source
# Description: Select from input/imported images to create a new batch of latent images, or select an empty latent.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng



##
# SYSTEM STATUS
##
print(f'\t - 🟢 🌱 Loaded FlowState Latent Source.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FlowState_Node import FlowState_Node


##
# OUTSIDE IMPORTS
##
import torch
import numpy as np
import hashlib

import os, sys
import node_helpers
import folder_paths

from PIL import Image, ImageOps, ImageSequence
from comfy import model_management


##
# NODES
##
class FlowState_LatentSource(FlowState_Node):
    CATEGORY = 'FlowState Creator Suite/Latent'
    DESCRIPTION = 'Create a new batch of latent images to be denoised via sampling.'
    FUNCTION = 'execute'
    RETURN_TYPES = TYPE_LATENT_SOURCE_OUT
    RETURN_NAMES = ('Latent Image',)
    OUTPUT_TOOLTIPS = ('The latent image batch.',)

    def __init__(self):
        super().__init__('🌊🌱 FlowState Latent Source')
        self.device = model_management.intermediate_device()
        self.latent_channels = 4

        self.have_input_image = False
        self.using_empty_latent = False
        self.using_input_image = False
        self.using_uploaded_image = False
        self.using_horizontal = False
        self.using_custom_resolution = False
        self.using_image_resolution = False

        self.empty_latent = None
        self.input_latent = None
        self.uploaded_latent = None
        self.uploaded_image = None

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
                'latent_type': TYPE_LATENT_SOURCE_INPUT_TYPE,
                'custom_width': TYPE_IMG_WIDTH,
                'custom_height': TYPE_IMG_HEIGHT,
                'custom_batch_size': TYPE_LATENT_BATCH_SIZE,
                'image': TYPE_INPUT_FILES(),
                'vae': TYPE_VAE_IN
            },
            'optional': {
                'input_img': TYPE_IMAGE
            }
        }

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

    def load_and_encode(self):
        image = self.batch_params['image']
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        latent = self.batch_params['vae'].encode(output_image[:,:,:,:3])
        self.uploaded_latent = latent
        self.uploaded_image = output_image

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
        self.using_uploaded_image = self.batch_params['latent_type'] == 'Uploaded Image'

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

        if self.using_uploaded_image:
            self.set_batch_out('latent from uploaded image', self.uploaded_latent)

    def execute(self, resolution, orientation, latent_type, custom_width, custom_height, custom_batch_size, image, vae, input_img=None):
        self.print_status([('Preparing latent batch...',)], init=True)

        batch_start_time = time.time()

        self.batch_params = locals()

        self.set_img_parameters()
        self.load_and_encode()
        self.prepare_latent_batch()

        batch_duration, batch_mins, batch_secs = get_mins_and_secs(batch_start_time)

        self.print_status([
            (f'Prepared {self.system_message}.', self.latent_batch_out['samples'].shape),
            ('Latent Batch Size', self.batch_params['custom_batch_size']),
            ('Preparation Time', f'{batch_mins}m {batch_secs}s ({batch_duration}s)'),
        ], end=True)

        return (self.latent_batch_out,)

