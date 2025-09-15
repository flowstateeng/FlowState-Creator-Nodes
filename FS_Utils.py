# Project: FlowState Utilities
# Description: Global utilities for all nodes.
# Author: Johnathan Chivington
# Contact: flowstateeng@gmail.com | youtube.com/@flowstateeng


##
# OUTSIDE IMPORTS
##
import os, sys, time
import folder_paths


# INFO
def get_mins_and_secs(start_time):
    duration = time.time() - start_time
    mins = int(duration // 60)
    secs = int(duration - mins * 60)
    return duration, mins, secs


# I/O
def get_input_files():
    input_dir = folder_paths.get_input_directory()
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files = folder_paths.filter_files_content_types(files, ["image"])
    return files

def get_output_files():
    output_dir = folder_paths.get_output_directory()
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    files = folder_paths.filter_files_content_types(files, ["image"])
    return files


# MODELS
def get_vae_list():
        vaes = folder_paths.get_filename_list('vae')
        approx_vaes = folder_paths.get_filename_list('vae_approx')
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith('taesd_decoder.'):
                sd1_taesd_dec = True
            elif v.startswith('taesd_encoder.'):
                sd1_taesd_enc = True
            elif v.startswith('taesdxl_decoder.'):
                sdxl_taesd_dec = True
            elif v.startswith('taesdxl_encoder.'):
                sdxl_taesd_enc = True
            elif v.startswith('taesd3_decoder.'):
                sd3_taesd_dec = True
            elif v.startswith('taesd3_encoder.'):
                sd3_taesd_enc = True
            elif v.startswith('taef1_encoder.'):
                f1_taesd_dec = True
            elif v.startswith('taef1_decoder.'):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append('taesd')
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append('taesdxl')
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append('taesd3')
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append('taef1')
        return vaes