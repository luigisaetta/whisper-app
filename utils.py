#
# Some utilities to work on audio
#
#
import pandas as pd
import soundfile as sf
from config import SAMPLE_RATE

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


def check_file(audio_path):
    if check_sample_rate(audio_path, ref_sample_rate=SAMPLE_RATE) == False:
        print(f"The sample rate is not {SAMPLE_RATE} Hz.")
    else:
        print(f"Sample rate is {SAMPLE_RATE} Hz, OK.")

    # check file is MONO
    if check_mono(audio_path) == False:
        print("The file is NOT MONO.")
    else:
        print("File is MONO, OK.")
        print()
