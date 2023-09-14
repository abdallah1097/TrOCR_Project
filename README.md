# TrOCR Detector
Submitted to blnk Egypt


[![N|Solid](https://media.licdn.com/dms/image/C4D0BAQFXhOLCPeWiXA/company-logo_200_200/0/1661947990889?e=1702512000&v=beta&t=mJ7HCHzYJ718TDWtHYwM4AYuPxGxVbD40k81dZpW-QQ)](blnk)

TrOCR is a Transformer-based OCR, This repo implements the TrOCR From scratch using Tensorflow, and Django. This README.md contains:

- Setting up environment settings
- See HTML in the right
- ✨Magic ✨


## Getting Started ...
Setting Up Environment: Create and Activate a Virtual Environment
```sh
python -m venv ocr_detector_venv
source ocr_detector_venv/Scripts/activate
```
Install required dependencies

```sh
pip install -r requirements.txt
```
Starting Django Server: Starting development server at http://127.0.0.1:8000/
```sh
python manage.py runserver
```

## Django App
Main interface app will be like this:

![Capture](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/58e67944-7263-4d48-a07d-b80a914495b8)

## Show PreProcessed App

This allows you to see how images are preprocessed (Before normalization).

![preprocessed_Images](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/015dd1e3-aa45-46eb-b7c4-48512cd532aa)

## Show Predict
First, we must upload the image:

![upload_image](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/c0e1bcfe-3a49-42d9-87b8-4a1679987e41)

Once you upload the image, we can predict:

![upload_successfully](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/22bae25d-a172-4f9c-9302-6a1b4f2e5917)

## Configure Model

Allows you to change parameters set in the config.py file:

![configuration_2](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/7fb9d062-5382-46c8-8b9a-a5989be32f6f)
![configuration_1](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/e7f6fe0a-a69e-4e96-823e-99cdde4f71de)

Once you change the configuration:
![config_succ](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/af318af7-1a3f-4145-b51a-32f43aa9f97c)

## See Training Logs

Starting Tensorboard:

![tensorboard](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/4aff1e85-0636-4543-95f1-2f5f7e36ebf6)


Logs Sample:

![tensorboard_logs](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/fb1cc593-97c3-4d08-bfc4-135124ebb773)
