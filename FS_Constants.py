# Project: FlowState Constants
# Description: Global constants for all nodes.
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng



##
# OUTSIDE IMPORTS
##
import importlib
import nodes


##
# CONSTANTS
##

# SYSTEM INFO
SYSTEM_NAME = 'FlowState Creator Suite'
SYSTEM_VERSION = '1.0.1'

# IMAGE PARAMETERS
MAX_RESOLUTION=16384

# TYPES
TOOLTIP_UNDERLINE = 32


# --- IMPORT KIJAI (THE GOAT) SAGE ATTENTION UNTIL COMFY CORE IMPLEMENTS A NODE
SAGE_ATTENTION_INSTALLED = False

KJNODES_INSTALLED = "PathchSageAttentionKJ" in nodes.NODE_CLASS_MAPPINGS

SAGE_AVAILABLE = False
SageAttention = None

try:
    importlib.import_module("sageattention")
    SAGE_ATTENTION_INSTALLED = True
except:
    SAGE_ATTENTION_INSTALLED = False

if SAGE_ATTENTION_INSTALLED and KJNODES_INSTALLED:
    SAGE_AVAILABLE = True
    SageAttention = nodes.NODE_CLASS_MAPPINGS['PathchSageAttentionKJ']()
