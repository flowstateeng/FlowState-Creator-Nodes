# Project: FlowState Chef Ingredients
# Description: Input your ingredients for the FlowState Chef node.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'\t - 🟢 🥗 Loaded FlowState Chef Ingredients.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FlowState_Node import FlowState_Node


##
# OUTSIDE IMPORTS
##
from comfy_extras.nodes_images import ImageStitch


##
# NODES
##
class FlowState_Chef_Ingredients(FlowState_Node):
    CATEGORY = 'FlowState Creator Suite/Alteration'
    DESCRIPTION = '👨‍🍳 Input your ingredients for the FlowState Chef node.'
    FUNCTION = 'execute'
    RETURN_TYPES = (TYPE_ANY,)
    RETURN_NAMES = ('chef_ingredients',)
    OUTPUT_TOOLTIPS = ('Ingredients to feed into the FlowState Chef node.',)

    def __init__(self):
        super().__init__('🌊🥗 FlowState Chef Ingredients')

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                # PROMPT SETTINGS
                'qwen_prompt_label': TYPE_FLOWSTATE_LABEL_PROMPT_CHEF_QWEN,
                'qwen_edit_prompt': TYPE_PROMPT_CHEF_QWEN,
                'flux_prompt_label': TYPE_FLOWSTATE_LABEL_PROMPT_CHEF_FLUX,
                'flux_refinement_prompt': TYPE_PROMPT_CHEF_FLUX,
                'image': TYPE_CHEF_IMAGE_IN
            },
            'optional': {
                'image_2': TYPE_CHEF_IMAGE_IN,
                'image_3': TYPE_CHEF_IMAGE_IN,
                'image_4': TYPE_CHEF_IMAGE_IN,
            }
        }
    
    def stitch_images(self):
        images = [
            self.params['image'],
            self.params['image_2'],
            self.params['image_3'],
            self.params['image_4']
        ]

        filtered = [image for image in images if image is not None]

        if not filtered:
            raise ValueError(self.format_value_error('ingredients_img'))

        self.stitched = filtered[0]

        for image in filtered[1:]:
            self.stitched = ImageStitch().stitch(
                image1=self.stitched,
                direction='right',
                match_image_size='true',
                spacing_width=0,
                spacing_color='white',
                image2=image
            )[0]

    def execute(self,
        qwen_prompt_label, qwen_edit_prompt, flux_prompt_label, flux_refinement_prompt,
        image, image_2=None, image_3=None, image_4=None
    ):

        self.print_status([('Preparing Chef Ingredients...',)], init=True)

        self.params = locals()
        self.stitch_images()

        self.print_status([('Ingredients Prepared.',)], end=True)

        return ({
            'qwen': qwen_edit_prompt,
            'flux': flux_refinement_prompt,
            'img': self.stitched
        }, )


