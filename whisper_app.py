#
# Whisper UI with Streamlit
#
# v 2.0: removed hack to load Whisper custom model
# added also compare mode, where it loads form a csv file the expected sentences
# that are shown on the right
#

import time
from PIL import Image
import pandas as pd
import streamlit as st
from annotated_text import annotated_text

# OpenAI codebase
# to normalize the expected sentence and easier comparison
from whisper import normalizers

# some check to make it more robust to human errors in tests
from utils import check_file

from config import APP_DIR, LOCAL_DIR, LOGO, COMPARE_MODE
from config import AUDIO_FORMAT_SUPPORTED, LANG_SUPPORTED
from config import TARGET_FILE, ENABLE_EXTRA_CONFIGS
from config import WHISPER_MODEL_SUPPORTED

# default for extra_configs
from config import TIOF_DEFAULT, NST_DEFAULT, LPT_DEFAULT, CRT_DEFAULT, TEMP_DEFAULT

from transcriber import Transcriber

# whisper model is loaded only once
# custom is: medium, HF fine tuned
# medium, large are vanilla Whisper models
# for custom, you must provide a FINE_TUNED_MODEL file
# in the dir where the app file is launched
whisper_models = WHISPER_MODEL_SUPPORTED

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


# compare and annotate transcription vs expected text
def compare(transcribed_text, expected_text):
    color = "#8ef"
    # tokenize to get individual words
    tokenized = transcribed_text.split(" ")

    new_word_with_annotation = []

    for word in tokenized:
        word = word.strip()
        if word not in expected_text:
            # annotate
            new_word_with_annotation.append((word + " ", "doubt", color))
        else:
            new_word_with_annotation.append(word + " ")

    return new_word_with_annotation


#
# to simplify settings of st.number_input
# see below extra_configs
#
def set_st_number_input(label, defaults):
    nb_input = st.number_input(
        label,
        min_value=defaults[0],
        max_value=defaults[1],
        value=defaults[2],
        step=defaults[3],
    )

    return nb_input


#
# to simplify settings of st.slider
#
def set_st_slider(label, defaults):
    st_slider = st.slider(
        label,
        min_value=defaults[0],
        max_value=defaults[1],
        value=defaults[2],
        step=defaults[3],
    )

    return st_slider


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
            # defaults are in config.py
            temperature = set_st_number_input("Temperature", TEMP_DEFAULT)

            temperature_increment_on_fallback = set_st_number_input(
                "Temperature Increment on Fallback", TIOF_DEFAULT
            )

            no_speech_threshold = set_st_slider("No Speech Threshold", NST_DEFAULT)

            logprob_threshold = set_st_slider("Logprob Threshold", LPT_DEFAULT)

            compression_ratio_threshold = set_st_slider(
                "Compression Ratio Threshold", CRT_DEFAULT
            )

            condition_on_previous_text = st.checkbox(
                "Condition on previous text", value=True
            )
    else:
        # setting defaults (be careful, changes can make it really worse)
        temperature = TEMP_DEFAULT[2]
        temperature_increment_on_fallback = TIOF_DEFAULT[2]
        no_speech_threshold = NST_DEFAULT[2]
        logprob_threshold = LPT_DEFAULT[2]
        compression_ratio_threshold = CRT_DEFAULT[2]
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
                # for comparison do the same normalization
                transcribed_txt = normalizer(transcriber.text)

                expected_txt = None
                try:
                    expected_txt = normalizer(target_dict[input_file.name])

                except:
                    # to handle the case not found
                    print(f"Expected text not found: {input_file.name}...")
                    print()

                # annotate word in transcribed but not in expected
                if expected_txt is not None:
                    word_annotated = compare(transcribed_txt, expected_txt)
                else:
                    word_annotated = [transcribed_txt]

                media_col.subheader("Transcribed vs expected text:")

                with media_col:
                    # we need the magical *
                    annotated_text(*word_annotated)

                media_col.write("")
                media_col.write("Expected text is: ")
                if expected_txt is not None:
                    media_col.write(expected_txt)

    else:
        st.error("Please upload a file!")
