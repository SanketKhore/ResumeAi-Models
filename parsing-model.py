# Step 1: Import and Train Resume Data Extraction Model
# This part of the code sets up the first model that extracts structured information from resume images.

import os
import json
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
## for step 2
import cv2
import numpy as np
import easyocr

# Load JSON exported from Label Studio
def load_label_studio_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for item in data:
        image_path = item['data']['ocr']  # Change if your label studio export uses a different key
        image_path = image_path.replace('/data/local-files/?d=', '')

        image = Image.open(image_path).convert('RGB')

        annotations = item['annotations'][0]['result']
        combined_text = {}
        for annotation in annotations:
            label = annotation['value']['labels'][0]
            text = annotation['value']['text']
            if label not in combined_text:
                combined_text[label] = []
            combined_text[label].append(text)

        for key in combined_text:
            combined_text[key] = ' '.join(combined_text[key])

        # Combine as a JSON string
        target_text = json.dumps(combined_text, indent=None)

        samples.append({
            'image': image,
            'text': target_text
        })

    return Dataset.from_list(samples)

# Load processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([processor.tokenizer.cls_token])[0]

# Preprocess function
def preprocess(example):
    pixel_values = processor(images=example['image'], return_tensors="pt").pixel_values.squeeze(0)
    decoder_input_ids = processor.tokenizer(example['text'], return_tensors="pt").input_ids.squeeze(0)
    return {
        'pixel_values': pixel_values,
        'labels': decoder_input_ids
    }

# Load and preprocess dataset
raw_dataset = load_label_studio_dataset("/path/to/label_studio_export.json")  # Replace path
processed_dataset = raw_dataset.map(preprocess)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./donut-resume-model",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    logging_dir="./logs",
    save_steps=500,
    evaluation_strategy="no",
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

# Define custom trainer
class DonutDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'pixel_values': item['pixel_values'],
            'labels': item['labels']
        }

train_dataset = DonutDataset(processed_dataset)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.tokenizer
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./donut-resume-model")
processor.save_pretrained("./donut-resume-model")




# Step 2: Formatting Analysis

def check_formatting(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Check for blur
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = lap_var < 100

    # 2. Read text with positions
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)

    font_sizes = []
    y_positions = []
    for (bbox, text, confidence) in results:
        (tl, tr, br, bl) = bbox
        height = np.linalg.norm(np.array(tl) - np.array(bl))
        y_positions.append(tl[1])
        font_sizes.append(height)

    spacing_consistency = np.std(np.diff(sorted(y_positions))) if len(y_positions) > 1 else 0
    font_size_var = np.std(font_sizes)

    return {
        "blur_score": lap_var,
        "is_blurry": is_blurry,
        "font_size_variation": font_size_var,
        "line_spacing_consistency": spacing_consistency
    }