# Whisper-App
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
This repository contains all the work I have done (and I'm doing) in developing a web app for **Speech-to-text**, based on **OpenAI Whisper**

## Updates
* 24/11/2022: **no need** anymore to change the Whisper codebase to load the custom model
* 25/11/2022: clean separation between frontend and backend

## Utility
* match_layers

One common use case could be that we're fine-tuning a Whisper model, for example to have higher accuracy on a special domain's language.
The fine tuning can be done using **HF Transformers**. 
In this case, the utility can be used to match and show how to load the custom tuned model in **Whisper codebase**.

## Libraries used
* Torch
* HF Transformers
* OpenAI Whisper
* Streamlit
* soundfile
* tqdm
* pickle
* pandas
* PIL




