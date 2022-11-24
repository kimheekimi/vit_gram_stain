# ViT for Gram-stain Image Classification

Table of contents:
- ViT for Gram-stain Image Classification
	- Installation
	- Models
	- Data
	- Bibtex
	- Changelog

## Installation
Codes can be seemlessly excuted in the docker container below.
Please pull the image of huggingface/optimum-onnxruntime:1.3.0-cuda and run a container from the image.
```
docker pull huggingface/optimum-onnxruntime:1.3.0-cuda
```

Go into the container and install addtional python packages
```
pip install pillow "optimum[onnxruntime-gpu]" evaluate[evaluator] sklearn mkl-include mkl --upgrade
```

## Overview


## Models


## Data
The approval of the Data Protection office is currently in the works. As soon as we get approval, we will add data to the GitHub repository and update the readme file accordingly. Meantime, we will provide the link to the DIBaS database which is a publicly accessible gram stain image dataset: https://github.com/gallardorafael/DIBaS-Dataset. 


## Bibtex
