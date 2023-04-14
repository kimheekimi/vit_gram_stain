'''
@author: Kim, Hee
EXAMPLE TO EXECUTE : python bin/evaluation.py -m MOBILEVIT -s MAX -d UMMDS -e 1
ref: https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTForImageClassification
'''
'''
Arguments
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', dest='model_name', choices=['BEIT','CONVNEXT','DEIT','MOBILEVIT','POOLFORMER','RESNET','SWIN','VIT'], required=True)
parser.add_argument('-s', '--model_size', dest='model_size', choices=['MIN','MAX'], required=True)
parser.add_argument('-d', '--', dest='dataset', choices=['UMMDS','DIBAS'], required=True)
parser.add_argument('-e', '--epoch', dest='epoch', required=True)
args = parser.parse_args()
'''
DATA LOAD
'''
from datasets import load_dataset 
data_dir = f"/workspace/gramdata/{args.dataset}/test/**"
ds_test = load_dataset("imagefolder", data_dir=data_dir, task="image-classification", split="train")
'''
CONVERT
'''
import os, logging, time
from optimum.onnxruntime import ORTModelForImageClassification
from pathlib import Path
from datasets import load_dataset
from transformers import AutoFeatureExtractor, pipeline
'''
BOILER PLATE
'''
work_dir = f"/workspace/gramstain/code/05_onnx"
model_path = Path(f"{work_dir}/model/{args.dataset}/{args.model_name}/{args.epoch}/{args.model_size}")
onnx_path  = Path(f"{model_path}/onnx/")
quant_path = Path(f"{onnx_path}/quant/")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, from_transformers=True)
from evaluate import evaluator
e = evaluator("image-classification")
'''
DEFIEN LOG
'''
logging.basicConfig(level=logging.INFO,
					format="%(asctime)s [%(levelname)s] %(message)s",
					handlers=[logging.FileHandler(f'{work_dir}/log/{args.dataset}/{args.model_name}_{args.model_size}_{args.epoch}_evaluate.txt'),
							  logging.StreamHandler()])
logger = logging.getLogger()
'''
PyTorch MODEL
'''
from transformers import AutoConfig, AutoModelForImageClassification
config = AutoConfig.from_pretrained(f"{model_path}/config.json")
model = AutoModelForImageClassification.from_pretrained(f"{model_path}/pytorch_model.bin", from_tf=False, config=config)
classifier = pipeline("image-classification", model=model, feature_extractor=feature_extractor)

results = e.compute(
    model_or_pipeline=classifier,
    data=ds_test,
    metric="f1",
    input_column="image",
    label_column="labels",
    label_mapping=model.config.label2id,
    strategy="simple",
)
logger.info(f"F1: {results['f1']*100:.2f}%")

start = time.time()
results = e.compute(
    model_or_pipeline=classifier,
    data=ds_test,
    metric="accuracy",
    input_column="image",
    label_column="labels",
    label_mapping=model.config.label2id,
    strategy="simple",
)
end = time.time()
logger.info(f"Elapsed Time: {end - start}")
logger.info(f"Accuracy: {results['accuracy']*100:.2f}%")
size = os.path.getsize(model_path/"pytorch_model.bin")/(1024*1024)
logger.info(f"Model Size: {size:.2f} MB")
'''
ONNX MODEL
'''
model = ORTModelForImageClassification.from_pretrained(model_path, from_transformers=True)
onnx_classifier = pipeline("image-classification", model=model, feature_extractor=feature_extractor)

results = e.compute(
    model_or_pipeline=onnx_classifier,
    data=ds_test,
    metric="f1",
    input_column="image",
    label_column="labels",
    label_mapping=model.config.label2id,
    strategy="simple",
)
logger.info(f"ONNX F1: {results['f1']*100:.2f}%")

start = time.time()
results = e.compute(
    model_or_pipeline=onnx_classifier,
    data=ds_test,
    metric="accuracy",
    input_column="image",
    label_column="labels",
    label_mapping=model.config.label2id,
    strategy="simple",
)
end = time.time()
logger.info(f"ONNX elapsed Time: {end - start}")
logger.info(f"ONNX Accuracy: {results['accuracy']*100:.2f}%")
size = os.path.getsize(onnx_path/"model.onnx")/(1024*1024)
logger.info(f"ONNX Size: {size:.2f} MB")

'''
QUANT MODEL EVALUATION
'''
qnt_model = ORTModelForImageClassification.from_pretrained(quant_path, file_name="model_quantized.onnx")
qnt_classifier = pipeline("image-classification", model=qnt_model, feature_extractor=feature_extractor)

results = e.compute(
    model_or_pipeline=qnt_classifier,
    data=ds_test,
    metric="f1",
    input_column="image",
    label_column="labels",
    label_mapping=model.config.label2id,
    strategy="simple",
)
logger.info(f"QNT F1: {results['f1']*100:.2f}%")

start = time.time()
results = e.compute(
    model_or_pipeline=qnt_classifier,
    data=ds_test,
    metric="accuracy",
    input_column="image",
    label_column="labels",
    label_mapping=qnt_model.config.label2id,
    strategy="simple",
)
end = time.time()
logger.info(f"QNT elapsed Time: {end - start}")
logger.info(f"QNT Accuracy: {results['accuracy']*100:.2f}%")
quantized_model = os.path.getsize(quant_path/"model_quantized.onnx")/(1024*1024)
logger.info(f"QNT Size: {quantized_model:.2f} MB")