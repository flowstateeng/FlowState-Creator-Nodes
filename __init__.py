# Project: FlowState Creator Suite Initialization
# Description: Initialize the FlowState Creator Suite modules.
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


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
import time



##
# SYSTEM STATUS
##
print(
    f'\n\n 🌊 FlowState Creator Suite 🌊'
    f'\n-------------------------------'
    f'\n  1. ⏳ System initializing...'
)


##
# CHECK SAGE ATTENTION
##
if KJNODES_INSTALLED:
    print('\t - 🟢 KJ Nodes available.')
else:
    print('\t - 🚨 KJNODES NOT AVAILABLE')


if SAGE_ATTENTION_INSTALLED:
    print('\t - 🟢 Sage Attention available.')
else:
    print('\t - 🚨 SAGE ATTENTION NOT AVAILABLE')


if SAGE_ATTENTION_INSTALLED and KJNODES_INSTALLED:
    print('\t   - ✅ KJ Nodes & Sage Attention available.')
    print('\t   - 🥳🎉 Activating Sage Attention!')
else:
    sage_but_not_kj = SAGE_ATTENTION_INSTALLED and not KJNODES_INSTALLED
    kj_but_not_sage = KJNODES_INSTALLED and not SAGE_ATTENTION_INSTALLED

    if sage_but_not_kj: print('\t   - ⛔ Sage Attention available, but KJ Nodes unavailable.')
    else: print('\t   - ⛔ KJ Nodes available, but Sage Attention unavailable.')

    print('\t   - 😭💔 Unable to activate Sage Attention for use in 🌊 FlowState Creator Suite.')



##
# LOAD NODES & MAPPINGS
##
load_start_time = time.time()

from .FS_Mappings import *


##
# SYSTEM STATUS
##
load_duration, load_mins, load_secs = get_mins_and_secs(load_start_time)

print(f'  4. 🚀 System fully loaded in {round(load_duration, 4)}s\n')

