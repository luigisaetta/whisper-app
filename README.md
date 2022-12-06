# Whisper-App
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository contains all the work I have done (and I'm doing) in developing a web app for **Speech-to-text**, based on **OpenAI Whisper**

## Updates
* 24/11/2022: **no need** anymore to change the Whisper codebase to load the custom model
* 25/11/2022: clean separation between frontend and backend

## Features
* You can load and use a **custom trained model**, using HF Transformers
* You can **enable comparison** of the transcription with **expected** text, providing a csv file (f_name, sentence)
* You can run on a GPU, and it is way faster
* supported models: medium, large (vanilla) and medium for custom

## Utility
* match_layers

One common use case could be that we're **fine-tuning** a Whisper model, for example to have higher accuracy on a special domain's language.

The fine tuning can be done using **HF Transformers**, using the approach described [here](https://huggingface.co/blog/fine-tune-whisper).

In this case, the utility can be used to match and show how to load the custom tuned model in **Whisper codebase**.

You can find some more information on this utility in the **Wiki**.

## Libraries used
* Torch
* HF Transformers
* OpenAI Whisper
* Streamlit
* st-annotated-text
* soundfile
* tqdm
* pickle
* pandas
* PIL

## Environment
* based on Python 3.10.6
* can be rebuilt using the provided requirements.txt

## Running on GPU
I have tested and the code works fine on a VM equipped with:
* NVIDIA GPU P100
* Ubuntu 22.04-2022.11.06
* Python 3.10

To enable the code to run on GPU you need only to set:
```
DEVICE = cuda 
```
in config file.

It is, obviously, much faster running on GPU, especially with long files (> 60 sec.)

In this table I report the results of two tests done, enabling and disabling the GPU:

| Test n. | Audio dur. in sec. | time on CPU (s.) | time on GPU (s.) |
| ------- | ------------- | ------------- | ------------- |
|       1 | 129 | 55  |   11 |
|       2 | 255 | 110 | 19.8 |

about 5 times faster!





