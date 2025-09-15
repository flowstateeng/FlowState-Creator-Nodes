# Project: FlowState Creator Nodes Initialization
# Description: Initialize the FlowState Creator Nodes modules.
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng



##
# SYSTEM STATUS
##
print(
    f'\n\n 🌊 FlowState Creator Nodes 🌊'
    f'\n-------------------------------'
    f'\n  1. ⏳ System initializing...'
)


##
# IMPORTS
##
import time
from .FS_Utils import get_mins_and_secs

load_start_time = time.time()

from .FS_Mappings import *


##
# SYSTEM STATUS
##
load_duration, load_mins, load_secs = get_mins_and_secs(load_start_time)

print(
    f'  4. 🚀 System fully loaded in {round(load_duration, 4)}s\n'
)
