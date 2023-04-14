'''
@author: Kim, Hee
EXAMPLE TO EXECUTE : python bin/finetuner.py -m RESNET -s MIN -d UMMDS -e 1
ref: https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTForImageClassification
'''
'''
BOILERPLATE
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', dest='model_name', choices=['BEIT','CONVNEXT','DEIT','MOBILEVIT','POOLFORMER','RESNET','SWIN','VIT'], required=True)
parser.add_argument('-s', '--model_size', dest='model_size', choices=['MIN','MAX'], required=True)
parser.add_argument('-d', '--dataset', dest='dataset', choices=['UMMDS','DIBAS'], required=True)
parser.add_argument('-e', '--epoch', dest='epoch', required=True)
args = parser.parse_args()
'''
DATA PREPARATION
'''
from datasets import load_dataset 
data_dir = f"/workspace/gramdata/{args.dataset}"
work_dir = f"/workspace/gramstain/code/05_onnx"
ds = load_dataset("imagefolder", data_dir=f"{data_dir}/train/**", split="train")
ds = ds.train_test_split(test_size=0.1, shuffle=True)
test_ds = load_dataset("imagefolder", data_dir=f"{data_dir}/test/**", split="train")
'''
MODEL PREPARATION
'''
model_name_or_path = ""
if "BEIT"==args.model_name:
    if args.model_size == "MIN": 
        model_name_or_path = "microsoft/beit-base-patch16-224"
    elif args.model_size == "MAX": 
        model_name_or_path = "microsoft/beit-large-patch16-224"
elif "CONVNEXT"==args.model_name:
    if args.model_size == "MIN": 
        model_name_or_path = "facebook/convnext-tiny-224"
    elif args.model_size == "MAX": 
        model_name_or_path = "facebook/convnext-large-224"
elif "DEIT"==args.model_name:
    if args.model_size == "MIN": 
        model_name_or_path = "facebook/deit-tiny-patch16-224"
    elif args.model_size == "MAX": 
        model_name_or_path = "facebook/deit-base-patch16-224"
elif "MOBILEVIT"==args.model_name:
    if args.model_size == "MIN": 
        model_name_or_path = "apple/mobilevit-xx-small"
    elif args.model_size == "MAX": 
        model_name_or_path = "apple/mobilevit-small"
elif "POOLFORMER"==args.model_name:
    if args.model_size == "MIN": 
        model_name_or_path = "sail/poolformer_s12"
    elif args.model_size == "MAX": 
        model_name_or_path = "sail/poolformer_m48"
elif "RESNET"==args.model_name:
    if args.model_size == "MIN": 
        model_name_or_path = "microsoft/resnet-18"
    elif args.model_size == "MAX": 
        model_name_or_path = "microsoft/resnet-152"
elif "SWIN"==args.model_name:
    if args.model_size == "MIN": 
        model_name_or_path = "microsoft/swin-tiny-patch4-window7-224"
    elif args.model_size == "MAX": 
        model_name_or_path = "microsoft/swin-large-patch4-window7-224"
elif "VIT"==args.model_name:
    if args.model_size == "MIN": 
        model_name_or_path = "google/vit-base-patch16-224"
    elif args.model_size == "MAX": 
        model_name_or_path = "google/vit-large-patch16-224"

from transformers import AutoImageProcessor
feature_extractor = AutoImageProcessor.from_pretrained(model_name_or_path, ignore_mismatched_sizes=True)
'''
DATA PREPROCESSING
'''
from torchvision.transforms import Resize, CenterCrop, Compose, Normalize, ToTensor

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
size = (
    feature_extractor.size["shortest_edge"]
    if "shortest_edge" in feature_extractor.size
    else (feature_extractor.size["height"], feature_extractor.size["width"])
)
_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples
	
ds = ds.with_transform(transforms)
test_ds = test_ds.with_transform(transforms)

from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()
'''
MATRICS
'''
import evaluate
accuracy = evaluate.load("accuracy")

import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
'''
TRAINING
'''
from transformers import AutoModelForImageClassification
labels = test_ds.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
	
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
model = AutoModelForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
	ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    f"{work_dir}/model/{args.dataset}/{args.model_name}/{args.epoch}/{args.model_size}",
    save_strategy="no",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=int(args.epoch),
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    logging_dir=f"{work_dir}/tensorboard/{args.dataset}/{args.model_name}/{args.epoch}/{args.model_size}",
    remove_unused_columns=False,
)    

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
'''
TEST
'''
metrics = trainer.evaluate(test_ds)
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)