# Whisper-App
This repository contains all the work I have done (and I'm doing) in developing a web app for **Speech-to-text**, based on **OpenAI Whisper**

## Utility
* match_layers

One common use case could be that we're fine-tuning a Whisper model, for example to have higher accuracy on a special domain's language.
The fine tuning can be done using **HF Transformers**. In this case, the utility can be use to match and show how to load the custom tuned model in **Whisper codebase**.

## Libraries used
* torch
* transformers
* whisper
* streamlit
* soundfile
* tqdm

# Hack
For now to load a custom-trained whisper model an hack is needed: I have changed the TODO <insert name>
