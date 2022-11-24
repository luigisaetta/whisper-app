# Whisper-App
This repository contains all the work I have done (and I'm doing) in developing a web app for **Speech-to-text**, based on **OpenAI Whisper**

## Utility
* match_layers

One common use case could be that we're fine-tuning a Whisper model, for example to have higher accuracy on a special domain's language.
The fine tuning can be done using **HF Transformers**. In this case, the utility can be used to match and show how to load the custom tuned model in **Whisper codebase**.

## Libraries used
* Torch
* HF Transformers
* OpenAI Whisper
* Streamlit
* soundfile
* tqdm

# Hack
For now to load a custom-trained whisper model an hack is needed: I have changed the file 
```
/Users/lsaetta/miniforge3/envs/whisper2/lib/python3.10/site-packages/whisper/__init__.py
```
that contains the function **load_model()**, where the actual loading of the Whisper model is done.

TODO: find a better way
