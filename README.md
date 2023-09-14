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


Dillinger is currently extended with the following plugins.
Instructions on how to use them in your own application are linked below.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantaneously see your updates!

Open your favorite Terminal and run these commands.

First Tab:

```sh
node app
```

Second Tab:

```sh
gulp watch
```

(optional) Third:

```sh
karma test
```

#### Building for source

For production release:

```sh
gulp build --prod
```

Generating pre-built zip archives for distribution:

```sh
gulp build dist --prod
```

## Docker

Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

```sh
cd dillinger
docker build -t <youruser>/dillinger:${package.json.version} .
```

This will create the dillinger image and pull in the necessary dependencies.
Be sure to swap out `${package.json.version}` with the actual
version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on
your host. In this example, we simply map port 8000 of the host to
port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart=always --cap-add=SYS_ADMIN --name=dillinger <youruser>/dillinger:${package.json.version}
```

> Note: `--capt-add=SYS-ADMIN` is required for PDF rendering.

Verify the deployment by navigating to your server address in
your preferred browser.

```sh
127.0.0.1:8000
```

## License

MIT

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
