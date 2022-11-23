#
# Whisper UI with Streamlit
# some inspiration from https://github.com/hayabhay/whisper-ui
#

import os
import time
from PIL import Image
import streamlit as st

# OpenAI codebase
import whisper

# some check to make it more robust to human errors in tests
from utils import check_sample_rate, check_mono, check_file

from config import APP_DIR, LOCAL_DIR
from transcriber import Transcriber

#
# functions
#

# whisper model is loaded only once
# limited to best performing models (not fine tuned till now)
whisper_models = ["medium", "large", "custom"]

@st.experimental_singleton
def get_whisper_model(model_name):
    assert model_name in whisper_models, "Model name not supported!"

    model = whisper.load_model(model_name)

    return model

# Set app wide config
st.set_page_config(
    page_title="Audio Transcription | Whisper UI",
    page_icon="🤖",
    layout="wide",
    menu_items={
        "Get Help": "https://luigisaetta.it",
        "Report a bug": "https://luigisaetta.it",
        "About": "This is a UI for OpenAI's Whisper.",
    },
)

# list of supported audio files
audio_supported = ["wav"]

# add a logo
image = Image.open(APP_DIR / "logo.png")

img_widg = st.sidebar.image(image)

# Render input type selection on the sidebar & the form
# removed link for now, only local files
input_type = st.sidebar.selectbox("Input Type", ["File"])

with st.sidebar.form("input_form"):
    if input_type == "Link":
        url = st.text_input("URL (video works fine)")
    elif input_type == "File":
        # for now only wav supported
        input_file = st.file_uploader("File", type=audio_supported)

    model_name = st.selectbox("Whisper model", options=whisper_models, index=1)
    
    extra_configs = st.expander("Extra Configs")
    with extra_configs:
        temperature = st.number_input(
            "Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1
        )
        temperature_increment_on_fallback = st.number_input(
            "Temperature Increment on Fallback",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.2,
        )
        no_speech_threshold = st.slider(
            "No Speech Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05
        )
        logprob_threshold = st.slider(
            "Logprob Threshold", min_value=-20.0, max_value=0.0, value=-1.0, step=0.1
        )
        compression_ratio_threshold = st.slider(
            "Compression Ratio Threshold",
            min_value=0.0,
            max_value=10.0,
            value=2.4,
            step=0.1,
        )
        condition_on_previous_text = st.checkbox(
            "Condition on previous text", value=True
        )

    language = st.selectbox("Language", options=["en", "it"], index=0)

    transcribe = st.form_submit_button(label="Transcribe")

if transcribe:
    print()
    print("Transcription in progress...")
    print()

    # load the whisper model
    whisper_model = get_whisper_model(model_name)

    if input_file:
        # first make a local copy of the file
        print("Making a local copy of input file...")

        audio_path = LOCAL_DIR / input_file.name

        with open(audio_path, "wb") as f:
            f.write(input_file.read())

        # check that sample rate and MONO is ok
        check_file(audio_path)

        # added language
        transcriber = Transcriber(audio_path, input_type, language)

        t_start = time.time()

        # Render transcriptions
        transcription_col, media_col = st.columns(2, gap="large")

        transcription_col.write("Transcription is in progress, please wait...")

        transcriber.transcribe(
            model_name,
            temperature,
            temperature_increment_on_fallback,
            no_speech_threshold,
            logprob_threshold,
            compression_ratio_threshold,
            condition_on_previous_text,
        )

        if transcriber:
            # Trim raw transcribed output off tokens to simplify
            raw_output = transcription_col.expander("Raw output")
            raw_output.write(transcriber.raw_output)

            # Show transcription in a nicer format
            for segment in transcriber.segments:
                transcription_col.markdown(
                    f"""[{round(segment["start"], 1)} - {round(segment["end"], 1)}] - {segment["text"]}"""
                )

            # add audio widget to enable to listen to audio
            transcription_col.audio(data=input_file)

            t_ela = round(time.time() - t_start, 1)

            print()
            print(f"Transcription end. Elapsed time: {t_ela} sec.")
            print()

    else:
        st.error("Please upload a file!")
