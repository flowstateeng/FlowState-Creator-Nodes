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

    def print_system_info(self):
        print(
            f'\n\n'
            f'\n 🌊 {self.system_name} ({self.system_version}) 🌊'
            f'\n---------------------------------------'
            f'\n\n'
        )

    def print_status(self, messages, init=False, end=False):
        if init:
            print(f'\n\n\n  --- STARTING {self.node_name} ---\n')

        print(f'\n {self.node_name}')

        for msg in messages:
            msg_str = f'  - {msg[0]}'

            if len(msg) > 1:
                msg_str += f': {msg[1]}'

            print(msg_str)

        print('\n')

        if end:
            print(f'\n\n --- {self.node_name} COMPLETE --- \n\n\n')
    
    def get_mins_and_secs(self, start_time):
        duration = time.time() - start_time
        mins = int(duration // 60)
        secs = int(duration - mins * 60)
        return round(duration, 4), mins, secs

