#
# Config file
# put all the relevant configurations here
#
import pathlib

APP_DIR = pathlib.Path(__file__).parent.absolute()

# this is a directory where a local copy of the wav is made
LOCAL_DIR = APP_DIR / "local"
LOCAL_DIR.mkdir(exist_ok=True)

# set to True if you have a NVIDIA GPU
# set to False on CPU to avoid warnings
FP16_MODE = False