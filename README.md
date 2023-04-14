# ViT for Gram-stain Image Classification

Table of contents:
- Lightweight Visual Transformer for  Gram- Stained Image Classification
	- Installation
	- Models
	- Data
	- Execution
	- Bibtex
	- Changelog

## Installation
Codes can be seemlessly excuted in the docker container below.
Please pull the image of pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel and run a container from the image.
```
docker pull pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel 
```

Go into the container and install addtional python packages
```
pip install pillow "optimum[onnxruntime-gpu]" evaluate[evaluator] sklearn mkl-include mkl --upgrade scikit-learn torchvision fairscale deepspeed transformers torch transformers[onnx] optimum[onnxruntime]
```

## Abstract
Gram-stain analysis is a laboratory procedure to detect the presence of microorganisms quickly for patients suffering from infectious diseases and we aimed to partially automate it. We already published a paper doing this with convolutional neural networks (CNNs) models, however, the applicability of emerging visual transformers (VT) was lacking. This study, therefore, investigated VT models under various configurations (parameters; epochs; and quantization schemes) on two datasets. Six distinctive VT models (BEiT, DeiT, MobileViT, PoolFormer, Swin and ViT) were examined by accuracy, inference time and model size and then they were compared with two CNNs models (ConvNeXT and ResNet). ViT attained the best accuracy (98.3%) and it consistently achieved competitive accuracies regardless of data and configuration settings. With regard to inference time, frames per second (FPS) of small models consistently surpassed that of large models by a substantial margin. The most agile VT model was DeiT small in int8 (6.0 FPS) in contrast to the slowest, which was BEiT large in float32 (1.4 FPS). The comprehensive model evaluation, considering the trade-offs between accuracy, inference time, and model size, was conducted by Friedman and Nemenyi test.It ranked DeiT as the  best model for Gram-stained image classification. 

## Models
Six distinctive VT models (BEiT, DeiT, MobileViT, PoolFormer, Swin and ViT) were examined by accuracy, inference time and model size and then they were compared with two CNNs models (ConvNeXT and ResNet).

## Data
The Ethics committee of Medical Faculty Mannheim, Heidelberg University allowed us to make data publicly available. We would like to thank the Institute for Clinical Chemistry and Institute of Medical Microbiology and Hygiene, Medical Faculty Mannheim of Heidelberg University, for providing the data. https://heibox.uni-heidelberg.de/d/6b672e3ff50a468191b9/

## Execution
- Fine-tuning to a custom dataset
```
python bin/finetuner.py --model_name RESNET --model_size MIN --dataset UMMDS --epoch 1
```
- Exporting a model to ONNX (ref: https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model)
```
optimum-cli export onnx --model MODEL_INPUT_PATH/ MODEL_OUTPUT_PATH/
```
- Quantizing an ONNX model (ref: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization)
```
 optimum-cli onnxruntime quantize --onnx_model MODEL_INPUT_PATH/ --avx512 -o MODEL_OUTPUT_PATH/
```
- Model evaluation on a test dataset
```
python bin/evaluation.py --model_name MOBILEVIT --model_size MAX --dataset UMMDS --epoch 1
```
## Bibtex
To be updated...

## Changelog
2023/04/14 - Initial commit
