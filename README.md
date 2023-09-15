# TrOCR Detector
Submitted to blnk Egypt


[![N|Solid](https://media.licdn.com/dms/image/C4D0BAQFXhOLCPeWiXA/company-logo_200_200/0/1661947990889?e=1702512000&v=beta&t=mJ7HCHzYJ718TDWtHYwM4AYuPxGxVbD40k81dZpW-QQ)](blnk)

TrOCR is a Transformer-based OCR, This repo implements the TrOCR From scratch using Tensorflow, and Django. This README.md contains:

![TrOCR](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/ca04e7b6-b529-49bd-aaa3-43056b5c2f0d)

The application is divided into two main apps:
1. TrOCR_Django_App: Main application containing the main webpage interface and communicates with Deep_Learning_App.
2. Deep_Learning_App: This application handles all deep learning implementation and scripts. For instance, building encoder, decorer, TrOCR Model, ```predict.py```, ```train.py``` and etc.


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
Edit network/ data configuration. If you prefer to use Vim:
```sh
vim Deep_Learning_App/src/config.py
```
or using Nano editor:
Edit network/ data configuration. If you prefer to use Vim:
```sh
nano Deep_Learning_App/src/config.py
```
Starting Django Server: Starting development server at http://127.0.0.1:8000/
```sh
python manage.py runserver
```

## Deep Learning App

Let's first see the implementation details of such a project:

1. Data Loader: This module:
    a. Loads the dataset images and corresponding text.
    b. Tokenize the words using [Bert Arabic Tokenizer](https://huggingface.co/asafaya/bert-base-arabic).
1. Preprocessing: The preprocessing implemented included:
    a. Extracting and cropping the text from the image: This part is done using [OpenCV Library](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) by thresholding and finding contours of text to extract  the text box as shown:

![268078097-015dd1e3-aa45-46eb-b7c4-48512cd532aa](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/6648700a-9cd2-4265-a542-9da26646731c)


    b. Image resizing after extracting the text: (88,200,3).
    c. Normalization: /255.0.
3. TrOCR Model: All encoder/ decoder architecture is written in [Tensorflow](https://www.tensorflow.org/) in OOP Well-documented Inhertided Classes. Encoder/ Decoder configuration parameters are to be edited in ```Deep_Learning_App/src/config.py```
## Run Deep Learning Scripts

You can train/ evaluate or predict without the need to use Django Apps. To do this you can train the model using:
```
cd TrOCR_Project
nano Deep_Learning_App/src/config.py
python Deep_Learning_App/src/train.py
```
Or to evaluate:
```
cd TrOCR_Project
nano Deep_Learning_App/src/config.py
python Deep_Learning_App/src/evaluate.py
```
Or to predict:
```
cd TrOCR_Project
nano Deep_Learning_App/src/config.py
python Deep_Learning_App/src/predict.py --image_path absolute/path/to/image.jpg
```

4. Loss/ Accuracy Masked Functions: Since Transformers require padding sequence length to have a unified length, predictions from paddings should not be accounted for loss/ accuracy calculations as they're being masked. Therefore, masked loss/ accuracy functions were created.
5. Learning Rate Scheduler: A custom learning rate scheduler according to the formula in the original Transformer was implemented:

![output_Xij3MwYVRAAS_1](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/41e9eeb3-7a60-4920-a01b-293ac8e002ef)
![Capture](https://github.com/abdallah1097/TrOCR_Project/assets/32100743/68c7911b-c14c-40ed-b45c-d95f9103e19d)

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


## Note

Kindly note for the sake of work-replication, small samples from images were added to the repo. For full replication, please add the dataset to: ```TrOCR_Project/media/dataset```
