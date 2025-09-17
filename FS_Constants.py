# Project: FlowState Constants
# Description: Global constants for all nodes.
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng



##
# OUTSIDE IMPORTS
##
import os, folder_paths, importlib
import nodes


##
# CONSTANTS
##
MAX_RESOLUTION=16384
TOOLTIP_UNDERLINE = 32


# --- IMPORT KIJAI (THE GOAT) SAGE ATTENTION UNTIL COMFY CORE IMPLEMENTS A NODE
SAGE_ATTENTION_INSTALLED = False

KJNODES_INSTALLED = "PathchSageAttentionKJ" in nodes.NODE_CLASS_MAPPINGS

SageAttention = None


try:
    importlib.import_module("sageattention")
    SAGE_ATTENTION_INSTALLED = True
except:
    SAGE_ATTENTION_INSTALLED = False

if SAGE_ATTENTION_INSTALLED and KJNODES_INSTALLED:
    SageAttention = nodes.NODE_CLASS_MAPPINGS['PathchSageAttentionKJ']()
