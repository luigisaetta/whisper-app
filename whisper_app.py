#
# Whisper UI with Streamlit
# some inspiration from https://github.com/hayabhay/whisper-ui
#
# v 2.0: removed hack to lad Whisper custom model
# added also compare mode, where it loads form a csv file the expected sentences
# that are shown on the right
#

import time
from PIL import Image
import pandas as pd
import streamlit as st

# OpenAI codebase
import whisper

# to normalize the expected sentence
from whisper import normalizers

# some check to make it more robust to human errors in tests
from utils import check_file

from config import APP_DIR, LOCAL_DIR, LOGO, COMPARE_MODE
from config import AUDIO_FORMAT_SUPPORTED, LANG_SUPPORTED
from config import TARGET_FILE, ENABLE_EXTRA_CONFIGS

from transcriber import Transcriber

# whisper model is loaded only once
# custom is: medium, HF fine tuned
# medium, large are vanilla Whisper models
# for custom, you must provide a FINE_TUNED_MODEL file
# in the dir where the app file is launched
whisper_models = ["custom", "medium", "large"]

#
# Here we load once the transcriber class and therefore the Whisper model
#
@st.experimental_singleton
def get_transcriber(model_name):
    assert model_name in whisper_models, "Model name not supported!"

    transcriber = Transcriber(model_name)

    return transcriber


def remove_path(f_name):
    f_name = f_name.split("/")[-1]

    return f_name


# here we load the csv file containing expected sentences
@st.experimental_singleton
def load_target_csv(f_name):
    # loads the file in a dict where the key is the wav file name
    # and the value the expected sentence
    df = pd.read_csv(f_name)

    # another column to remove the path and keep only file name
    df["f_name"] = df["path"].apply(remove_path)

    target_dict = {}

    for k, v in zip(list(df["f_name"].values), list(df["sentence"].values)):
        target_dict[k] = v

    return target_dict

# Whisper English normalizer
@st.experimental_singleton
def get_normalizer():
    normalizer = normalizers.EnglishTextNormalizer()

    return normalizer


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
audio_supported = AUDIO_FORMAT_SUPPORTED

# add a logo
image = Image.open(APP_DIR / LOGO)
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

    # to choose the model
    model_name = st.selectbox("Whisper model", options=whisper_models, index=0)

    # normally we don't want to enable changes of these params
    # but you can enable in config.py
    if ENABLE_EXTRA_CONFIGS:
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
                "No Speech Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
            )
            logprob_threshold = st.slider(
                "Logprob Threshold",
                min_value=-20.0,
                max_value=0.0,
                value=-1.0,
                step=0.1,
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
    else:
        # setting defaults (be careful, changes can make it really worse)
        temperature = 0.0
        temperature_increment_on_fallback = 0.2
        no_speech_threshold = 0.6
        logprob_threshold = -1.0
        compression_ratio_threshold = 2.4
        condition_on_previous_text = True

    language = st.selectbox("Language", options=LANG_SUPPORTED, index=0)

    compare_mode = st.radio(label="Show target", options=["Yes", "No"], horizontal=True)

    transcribe = st.form_submit_button(label="Transcribe")

if transcribe:
    # load the whisper model (only the first time)
    transcriber = get_transcriber(model_name)

    if COMPARE_MODE:
        target_dict = load_target_csv(TARGET_FILE)
        normalizer = get_normalizer()

    # Render transcriptions
    # transcription_col, media_col = st.columns(2, gap="large")
    # 2:1 ration of transcrition col with compared to media col
    transcription_col, media_col = st.columns(gap="large", spec=[1, 1])

    if input_file:
        with st.spinner("Transcription in progress..."):

            t_start = time.time()

            # first make a local copy of the file
            print("Making a local copy of input file...")

            audio_path = LOCAL_DIR / input_file.name

            with open(audio_path, "wb") as f:
                f.write(input_file.read())

            # check that sample rate and MONO is ok
            print("Doing some checks on the audio file...")
            check_file(audio_path)

            transcription_col.subheader("The transcription:")

            # here we ask for transcription
            # it populate transcriber state
            transcriber.transcribe(
                audio_path,
                language,
                temperature,
                temperature_increment_on_fallback,
                no_speech_threshold,
                logprob_threshold,
                compression_ratio_threshold,
                condition_on_previous_text,
            )

            # Show transcription in a nicer format
            for segment in transcriber.segments:
                transcription_col.markdown(
                    f"""[{round(segment["start"], 1)} - {round(segment["end"], 1)}] - {segment["text"]}"""
                )

            # add audio widget to enable to listen to audio
            transcription_col.audio(data=input_file)

            # Trim raw transcribed output off tokens to simplify
            raw_output = transcription_col.expander("Raw output")
            raw_output.write(transcriber.raw_output)

            t_ela = round(time.time() - t_start, 1)

            print()
            print(f"Transcription end. Elapsed time: {t_ela} sec.")
            print()

            # if we want to show also
            # the expected text
            if compare_mode == "Yes":
                media_col.subheader("The expected text:")
                media_col.write(normalizer(target_dict[input_file.name]))

    else:
        st.error("Please upload a file!")
