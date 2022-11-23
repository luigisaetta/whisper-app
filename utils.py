#
# Some utilitis to work on audio
#
#
import pandas as pd
import os
import soundfile as sf
import re
import torch
import glob


#
# check sample rate... normally we're working with 16000
#
def check_sample_rate(file_wav_name, ref_sample_rate=16000):
    is_ok = True
    data, sample_rate = sf.read(file_wav_name)
    
    if sample_rate != ref_sample_rate:
        is_ok = False
            

    return is_ok

def get_audio_channels(path_name):
    info_obj = sf.info(path_name)
    
    return info_obj.channels

def check_mono(file_wav_name):
    MONO = 1
    is_ok = True
    
    num_channels = get_audio_channels(file_wav_name)
    
    if num_channels != MONO:
        is_ok = False
    
    return is_ok