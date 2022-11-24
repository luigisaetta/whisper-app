#
# Author: L. Saetta (2022)
# This utility has been used to create and save the map_dict
# used to load a custom-trained model into the Whisper codebase
#
# for now, tested only with "medium" model
# and, it works only for multi-lingual (no .en) models
#
import whisper
from transformers import WhisperForConditionalGeneration
import torch
from tqdm import tqdm

# using pickle to serialize the map_dict
import pickle

#
# Configuration
#
# set to True if the custom model has been trained using DDP (multi-gpu)
# as in my case, in the custom HF model, keys have a prefix (model.)
# it should come from the fact that I have trained on a milti-gpu machine, using DDP
DDP_TRAINED = True
# for now, tested only with medium
MODEL_SIZE = "medium"

# the device where you're running this code
DEVICE = "cpu"

# the name of the file with your fine-tuned model
FINE_TUNED_MODEL = "medium-custom.pt"

# the name of the file for the serialized map_dict
FILE_DICT = "map_dict.pkl"

#
# functions
#
# get if it is encoder or decoder
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


# the following 3 func have been added to make more check on the matching layers
# I check that are both from the same module (encoder or decoder)
# and with the same number


def extract_function(key_name):
    # encoder or decoder is the first part of the key
    first_part = key_name.split(".")[0]

    key_func = None
    if first_part in ["enconder", "decoder"]:
        key_func = first_part

    return key_func


def extract_layer_num(key_name):
    # layer num is the third piece
    layer_num = None

    if has_numbers(key_name):
        layer_num = key_name.split(".")[2]

    return layer_num


def sanity_check(key1, key2):
    # check same func (encoder or decoder)
    func1 = extract_function(key1)
    func2 = extract_function(key2)

    if func1 != func2:
        print(f"Warning: layers seem to have different functions: {key1},{key2}")

    # check same layer_num
    layer1 = extract_layer_num(key1)
    layer2 = extract_layer_num(key2)

    if layer1 != layer2:
        print(f"Warning: layers seem to have different numbers: {key1},{key2}")


#
# Main
#
if DDP_TRAINED:
    PREFIX = "model."
else:
    PREFIX = ""

print()
print("Loading vanilla Whisper model")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)

print("Loading vanilla HF Model")
hugging_face_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-" + MODEL_SIZE
)

# extract state-dict from both
state_d_standard = model.state_dict()
state_d_huggingface = hugging_face_model.model.state_dict()

# build the mapping between keys...
map_dict = {}
print("Matching layers...")

for k in tqdm(state_d_standard):
    for j in state_d_huggingface:
        if state_d_huggingface[j].shape == state_d_standard[k].shape:
            if torch.all(torch.eq(state_d_huggingface[j], state_d_standard[k])).item():
                map_dict[k] = j
                # make some check and eventually print a warning
                sanity_check(k, j)
                break

# check if we have matched every entry
print(f"Number of keys: {len(map_dict.keys())}")
print()
assert len(map_dict.keys()) == len(
    state_d_standard.keys()
), "The match is not complete !"

print("Match is complete !!!")
print()

# serialize the map_dict to file
print("Serializing map_dict...")
print()

with open(FILE_DICT, "wb") as f:
    pickle.dump(map_dict, f)
    f.close()

#
# In this section we do a test to see if the model can be actually loaded
#
print("Test if it works...")
print()

# loading with match keys
# restart from pickle file
print("Reloading map_dict...")
print()
with open(FILE_DICT, "rb") as f:
    map_dict = pickle.load(f)

# loading fine-tuned dict
print("Loading fine tuned dict...")
# added map_location to handle the fact that the custom model has been trained on GPU
state_dict_finetuned = torch.load(FINE_TUNED_MODEL, map_location=torch.device(DEVICE))

# build the state_dict to be used
# take the key name from standard (OpenAI) and the value from finetuned (HF)
print("Rebuild the state dict...")
new_state_dict = {}
n_except = 0
for k in tqdm(map_dict.keys()):
    try:
        # must add "model." because I come from DDP
        new_state_dict[k] = state_dict_finetuned[PREFIX + map_dict[k]]
    except:
        n_except += 1

assert n_except == 0, "Rebuild state dict failed"

print()
print("Loading the final model...")
model.load_state_dict(new_state_dict)

print("Loading ok")
print()

# print(model)
