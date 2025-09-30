# Project: FlowState Node Base Class
# Description: Base node that all FlowState Nodes inherit from, so that they have access to global data and methods.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# OUTSIDE IMPORTS
##
import os, sys, time


##
# FS IMPORTS
##
from .FS_Constants import *


##
# BASE FS NODE
##
class FlowState_Node:
    def __init__(self, node_name='FlowState Creator Suite Node: MISSING NAME'):
        self.system_name = SYSTEM_NAME
        self.system_version = SYSTEM_VERSION
        self.node_name = node_name

        self.errors = {
            'checkpoint': [
                ('Error loading model. Make sure whether you are loading a checkpoint or not.', ),
                ('Be sure to select the right "model_filetype" for the model you are selecting.', )
            ],
            'passthrough': [
                ('Error loading model. Cannot use combination of loaded & passthrough models.', ),
                ('Be sure to either LOAD the model, CLIP & VAE -- OR -- PASS THROUGH the model, CLIP & VAE.', )
            ],
            'ingredients_img': [
                ('FlowState Chef Ingredients requires at least one image input.', )
            ]
        }

    def format_value_error(self, error_type):
        msg = self.errors[error_type]

        formatted_msg = [self.node_name] + [line[0] for line in msg]

        formatted_msg = '\n'.join(formatted_msg)

        self.print_status(msg, error=True)

        return formatted_msg

    def print_system_info(self):
        print(
            f'\n\n'
            f'\n 🌊 {self.system_name} ({self.system_version}) 🌊'
            f'\n---------------------------------------'
            f'\n\n'
        )

    def print_status(self, messages, init=False, end=False, error=False):
        if init:
            print(f'\n\n\n  --- STARTING {self.node_name} ---')

        print(f'\n\n')

        if error: print('-' * 100)

        print(self.node_name)

        for msg in messages:
            msg_str = f'  - {msg[0]}'

            if len(msg) > 1:
                msg_str += f': {msg[1]}'

            print(msg_str)
        
        if error: print('-' * 100)

        print(' ')

        if end:
            print(f'\n --- {self.node_name} COMPLETE --- \n\n\n')
    
    def get_mins_and_secs(self, start_time):
        duration = time.time() - start_time
        mins = int(duration // 60)
        secs = int(duration - mins * 60)
        return round(duration, 4), mins, secs

