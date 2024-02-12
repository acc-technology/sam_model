#using dino to annoted open dataset
from groundingdino.util.inference import load_model, load_image, predict, annotate
import os
import supervision as sv
import cv2
import numpy as np


from ultralytics import YOLO
from PIL import Image
import torch

yolo = YOLO('yolov8n.pt')
yolo.train(data='/home/lcj/lab/else/sam_model/blind_data_split/data.yaml', epochs=50)
valid_results = yolo.val()
print(valid_results)

def run_yolo(yolo, image_url, conf=0.25, iou=0.7):
    results = yolo(image_url, conf=conf, iou=iou)
    res = results[0].plot()[:, :, [2,1,0]]
    return Image.fromarray(res)


# yolo = YOLO('runs/detect/train/weights/best.pt')

# image_url = 'test-01.jpg'
# predict(image_url)  
