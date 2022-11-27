"""
Module that handle a transcription
The class also handles correctly the loading of a fine-tuned model (HF)
see also match_layers

v2: more clean separation between frontend and backend
"""

import numpy as np
import torch
import pickle
from tqdm import tqdm
import whisper

from config import FILE_DICT, FINE_TUNED_MODEL, DEVICE
from config import FP16_MODE, PREFIX


class Transcriber:
    # removed start and duration, simplified

    def __init__(self, model_name):
        # disable/enable eventually fp16 mode (to avoid warnings on cpu)
        # read in config.py
        self.fp16 = FP16_MODE

        # load and store the whisper model
        self.model = self.get_whisper_model(model_name)

    # needed for custom models
    # used by get_whisper_model()
    def rebuild_state_dict(self, prefix, map_dict, state_dict_finetuned):
        # prefix could be model. or empty
        print("Rebuild the state dict...")

        new_state_dict = {}
        n_except = 0
        for k in tqdm(map_dict.keys()):
            try:
                # must add "model." because I come from DDP
                new_state_dict[k] = state_dict_finetuned[prefix + map_dict[k]]
            except:
                n_except += 1

        assert n_except == 0, "Rebuild state dict failed"

        return new_state_dict

    # load the Whisper model
    def get_whisper_model(self, model_name):
        model = None

        if model_name != "custom":
            # vanilla model
            print("Loading vanilla Whisper model...")
            model = whisper.load_model(model_name, device=DEVICE)
        else:
            # handle here custom (fine-tuned) model loading

            # this is needed to get Dims correctly
            print("Loading vanilla Whisper model...")
            model = whisper.load_model("medium", device=DEVICE)

            print("Loading map_dict...")
            print()
            with open(FILE_DICT, "rb") as f:
                map_dict = pickle.load(f)

            # loading fine-tuned dict
            print("Loading fine tuned dict...")
            # here we load the file containing the state of the fine-tuned model
            # added map_location to handle the fact that the custom model has been trained on GPU
            state_dict_finetuned = torch.load(
                FINE_TUNED_MODEL, map_location=torch.device(DEVICE)
            )

            # build the new state_dict to be used
            # take the key name from standard (OpenAI) and the value from finetuned (HF)
            new_state_dict = self.rebuild_state_dict(
                PREFIX, map_dict, state_dict_finetuned
            )

            print()
            print("Loading the fine tuned model state...")
            model.load_state_dict(new_state_dict)
            print()

            # modified here
            model = model.to(DEVICE)

        return model

    #
    # this one produces the transcription
    #
    def transcribe(
        self,
        audio_path,
        language: str,
        temperature: float,
        temperature_increment_on_fallback: float,
        no_speech_threshold: float,
        logprob_threshold: float,
        compression_ratio_threshold: float,
        condition_on_previous_text: bool,
    ):

        # Set configs & transcribe
        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        self.raw_output = self.model.transcribe(
            str(audio_path.resolve()),
            temperature=temperature,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            verbose=True,
            # added (LS)
            language=language,
            fp16=self.fp16,
            # added 25/11/2022
            task="transcribe",
        )

        # For simpler access
        self.text = self.raw_output["text"]
        self.language = self.raw_output["language"]
        self.segments = self.raw_output["segments"]

        # Remove token ids from the output
        for segment in self.segments:
            del segment["tokens"]

        self.transcribed = True
