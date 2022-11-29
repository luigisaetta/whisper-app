#
# Config file
# put all the relevant configurations here
#
import pathlib

APP_DIR = pathlib.Path(__file__).parent.absolute()

# this is a directory where a local copy of the wav is made
LOCAL_DIR = APP_DIR / "local"
LOCAL_DIR.mkdir(exist_ok=True)

LOGO = "logo.png"

AUDIO_FORMAT_SUPPORTED = ["wav"]
LANG_SUPPORTED = ["en", "it"]

# could be cpu or cuda (or mps?)
DEVICE = "cpu"

# set to True if you have a NVIDIA GPU
# set to False on CPU to avoid warnings
FP16_MODE = False

# for custom models:

# prefix to be eventually added to key in rebuild_state_dict
# this one is if the custom model has been traned with gpu
PREFIX = "model."

# the name of the file for the serialized map_dict
FILE_DICT = "map_dict.pkl"
# the name of the file with your fine-tuned model
FINE_TUNED_MODEL = "medium-custom.pt"

# sample_rate expected in Hz
SAMPLE_RATE = 16000

# to enable comparison with target
COMPARE_MODE = True

# this file contains the expected sentences
TARGET_FILE = "atco2.csv"

# to show the section for setting extra configs
ENABLE_EXTRA_CONFIGS = False

# default for EXTRA_CONFIGS

# temperature defaults
TEMP_DEFAULT = [0.0, 1.0, 0.0, 0.1]

# temperature_increment_on_fallback
TIOF_DEFAULT = [0.0, 1.0, 0.2, 0.2]

# no_speech_threshold
NST_DEFAULT = [0.0, 1.0, 0.6, 0.05]

# logprob_threshold
LPT_DEFAULT = [-20.0, 0.0, -1.0, 0.1]

# compression_ratio_threshold
CRT_DEFAULT = [0.0, 10.0, 2.4, 0.1]

