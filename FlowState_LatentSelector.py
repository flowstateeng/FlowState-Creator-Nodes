# Project: FlowState Latent Selector
# Description: Select from input/imported images to create a new batch of latent images, or select an empty latent.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng



##
# SYSTEM STATUS
##
print(f'\t - ✅ Loaded Latent Selector')


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
class FlowState_LatentSelector:
    CATEGORY = 'FlowState/latent'
    DESCRIPTION = 'Create a new batch of latent images to be denoised via sampling.'
    FUNCTION = 'execute'
    RETURN_TYPES = TYPE_LATENT_SELECTOR_OUT
    RETURN_NAMES = ('Latent Image', )
    OUTPUT_TOOLTIPS = (
        'The latent image batch.',
        'The image batch.',
        'Image width.',
        'Image height.',
    )

    @classmethod
    def __init__(self):
        self.device = model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'resolution': TYPE_LATENT_SELECTOR_RESOLUTION,
                'orientation': TYPE_LATENT_SELECTOR_ORIENTATION,
                'latent_type': TYPE_LATENT_SELECTOR_INPUT_TYPE,
                'custom_width': TYPE_IMG_WIDTH,
                'custom_height': TYPE_IMG_HEIGHT,
                'custom_batch_size': TYPE_LATENT_BATCH_SIZE,
                'image': TYPE_INPUT_FILES,
                'vae': TYPE_VAE_IN
            },
            'optional': {
                'input_image': TYPE_IMAGE
            }
        }

    @classmethod
    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return latent

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

    @classmethod
    def load_and_encode(self, image, vae):
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

        latent = vae.encode(output_image[:,:,:,:3])
        return latent, output_image

    def prepare_empty_latent(self, resolution, orientation, custom_width, custom_height, custom_batch_size):
        print(f'  - Preparing empty latent.\n')

        horizontal_img = orientation == 'Horizontal'

        width_to_use = custom_width
        height_to_use = custom_height

        if resolution != 'Custom':
            res_split = resolution.split(' - ')[0].split('x')
            width_to_use = int(res_split[0] if horizontal_img else res_split[1])
            height_to_use = int(res_split[1] if horizontal_img else res_split[0])

        generated_latent = self.generate(width_to_use, height_to_use, custom_batch_size)
        return ({'samples': generated_latent}, )

    def execute(self, resolution, orientation, latent_type, custom_width, custom_height, custom_batch_size, image, vae, input_img=None):
        print(f'\n\n\nFlowState Latent Selector')

        loaded_latent, loaded_image = self.load_and_encode(image, vae)
        have_pixels = input_img != None

        if latent_type == 'Empty Latent':
            return self.prepare_empty_latent(resolution, orientation, custom_width, custom_height, custom_batch_size)

        if latent_type == 'Uploaded Image':
            print(f'  - Preparing latent from loaded image.')
            return ({'samples': loaded_latent}, )
        
        if latent_type == 'Input Image':
            if have_pixels:
                print(f'  - Preparing latent from input image.')
                input_latent = vae.encode(input_img[:,:,:,:3])
                return ({'samples': input_latent}, )
            else:
                print(f'  - No input image.')
                return self.prepare_empty_latent(resolution, orientation, custom_width, custom_height, custom_batch_size)
