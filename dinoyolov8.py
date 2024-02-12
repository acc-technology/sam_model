#using dino to annoted open dataset
from groundingdino.util.inference import load_model, load_image, predict, annotate
import os
import supervision as sv
import cv2
import numpy as np

def run_dino(dino, image, text_prompt='food', box_threshold=0.4, text_threshold=0.1):
    boxes, logits, phrases = predict(
        model = dino, 
        image = image, 
        caption = text_prompt, 
        box_threshold = box_threshold, 
        text_threshold = text_threshold
    )
    return boxes, logits, phrases

dino = load_model('../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', '../GroundingDINO/weights/groundingdino_swint_ogc.pth')

# os.system('wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg')
image_source, image = load_image('dog.jpeg')
boxes, logits, phrases = run_dino(dino, image, text_prompt='dog')

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# sv.plot_image(annotated_frame, (8, 8))

def save_annotated_image(image_source, boxes, output_path):
    # 将图像源转换为OpenCV格式
    image_cv2 = image_source

    # 在图像上绘制标注框
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # 将带有标注框的图像保存到本地
    cv2.imwrite(output_path, image_cv2)

# 保存带有标注框的图像
save_annotated_image(annotated_frame, boxes, "annotated_image.jpg")

from datasets import load_dataset
import yaml
from tqdm import tqdm

def annotate(dino, data, data_size, data_dir):
    data = data.train_test_split(train_size=min(len(data), data_size))['train']

    image_dir = f'{data_dir}/images'
    label_dir = f'{data_dir}/labels'
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    for i, d in enumerate(tqdm(data)):
        image_path = f'{image_dir}/{i:06d}.png'
        label_path = f'{label_dir}/{i:06d}.txt'
        image = d['image'].resize((640, 640))
        image.save(image_path)
        
        image_source, image = load_image(image_path)
        boxes, logits, phrases = run_dino(dino, image)

        label = ['0 ' + ' '.join(list(map(str, b))) for b in boxes.tolist()]
        label = '\n'.join(label)
        with open(label_path, 'w') as f:
            f.write(label)


data = load_dataset('food101')
# annotate(dino, data['train'], 1000, 'data/train')
# annotate(dino, data['validation'], 200, 'data/valid')

config = {
    'names': ['food'],
    'nc': 1,
    'train': 'train/images',
    'val': 'valid/images'
}

with open('data/data.yaml', 'w') as f:
    yaml.dump(config, f)

from ultralytics import YOLO
from PIL import Image
import torch

yolo = YOLO('yolov8n.pt')
yolo.train(data='/home/lcj/lab/else/sam_model/data/data.yaml', epochs=5)
valid_results = yolo.val()
print(valid_results)

def run_yolo(yolo, image_url, conf=0.25, iou=0.7):
    results = yolo(image_url, conf=conf, iou=iou)
    res = results[0].plot()[:, :, [2,1,0]]
    return Image.fromarray(res)


yolo = YOLO('runs/detect/train/weights/best.pt')

image_url = 'test-01.jpg'
predict(image_url)  
