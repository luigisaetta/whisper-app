"""
Module that handle a transcription

from https://github.com/hayabhay/whisper-ui
"""
from typing import Union

import numpy as np
import torch
import whisper

from config import FP16_MODE


class Transcriber:
    # removed start and duration
    # simplified
    def __init__(self, audio_path, source_type: str, language: str):
        # this is now the path to a local copy
        self.audio_path = audio_path
        self.source_type = source_type
        self.transcribed = False
        # to force to english or it language only
        self.language = language

        # enable eventually fp16 mode (to avoid warnings on cpu)
        # read in config.py
        self.fp16 = FP16_MODE

    def transcribe(
        self,
        whisper_model: str,
        temperature: float,
        temperature_increment_on_fallback: float,
        no_speech_threshold: float,
        logprob_threshold: float,
        compression_ratio_threshold: float,
        condition_on_previous_text: bool,
        keep_model_in_memory: bool = True,
    ):

        # Get whisper model
        transcriber = whisper.load_model(whisper_model)

        # Set configs & transcribe
        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        self.raw_output = transcriber.transcribe(
            str(self.audio_path.resolve()),
            temperature=temperature,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            verbose=True,
            # added (LS)
            language=self.language,
            fp16=self.fp16,
        )

        # For simpler access
        self.text = self.raw_output["text"]
        self.language = self.raw_output["language"]
        self.segments = self.raw_output["segments"]

        # Remove token ids from the output
        for segment in self.segments:
            del segment["tokens"]

        self.transcribed = True

        if not keep_model_in_memory:
            del transcriber
            torch.cuda.empty_cache()
