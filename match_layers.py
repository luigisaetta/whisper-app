
#
# Author: L. Saetta (2022)
# This utility has been used to create and save the map_dict
# used to load a custom-trained model into the Whisper codebase
#
import whisper
from transformers import WhisperForConditionalGeneration
import torch
from tqdm import tqdm

# using pickle to serialize the map_dict
import pickle

print()
print("Loading vanilla Whisper model")
model = whisper.load_model("medium", device="cpu")

print("Loading vanilla HF Model")
hugging_face_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

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
                break

# check if we have matched every entry
print(f"Number of keys: {len(map_dict.keys())}")
print()
assert len(map_dict.keys()) == len(state_d_standard.keys()), "The match is not complete !"

print()
print("Match is complete !!!")
print()

# serialize the map_dict to file
print("Serializing map_dict...")
print()
FILE_DICT = "map_dict.pkl"
with open(FILE_DICT, 'wb') as f:
    pickle.dump(map_dict, f)
    f.close()

# loading with match keys
# restart from pickle file
print("Reloading map_dict...")
print()
with open(FILE_DICT, 'rb') as f:
    map_dict = pickle.load(f)

# loading fine-tuned dict
print("Loading fine tuned dict...")
FINE_TUNED_MODEL = "medium-custom.pt"
state_dict_finetuned = torch.load(FINE_TUNED_MODEL, map_location=torch.device('cpu'))

# build the state_dict to be used
# take the key name from standard (OpenAI) and the value from finetuned (HF)
print("Rebuild the state dict...")
new_state_dict = {}
n_except = 0
for k in tqdm(map_dict.keys()):
    try:
        # must add "model." because I come from DDP
        new_state_dict[k] = state_dict_finetuned["model." + map_dict[k]]
    except:
        n_except += 1

assert n_except == 0, "Rebuild state dict failed"

print()
print("Loading the final model...")
model.load_state_dict(new_state_dict)

print("Loading ok")

# print(model)
