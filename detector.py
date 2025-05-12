import os
import sys
import time
import torch
import logging
import torchvision
import ultralytics
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from loader import DataLoader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Detector")

CLASSS_LABELS = ['UAV', 'Airliner', 'Balloon', 'Bird', 'Helicopter']
models = {
    'fasterrcnn': 'fasterrcnn.pth',
    'yolov11': 'yolov11.pt',
}

class Detector:
    def __init__ (self, modelName, confidenceThreshold=0.6):
        self.modelName = modelName
        self.confidenceThreshold = confidenceThreshold
        self.modelPath = os.path.join('models', models[modelName])
        if not os.path.exists(self.modelPath):
            log.error(f"Model file {self.modelPath} not found.")
            sys.exit(1)
        log.info(f"Model file found: {self.modelPath}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.loadModel()
           
    def loadModel(self):
        if self.modelName == 'fasterrcnn':
            # Load the fasterrcnn mode
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None, num_classes=len(CLASSS_LABELS)+1
            )
            checkpoint = torch.load(self.modelPath, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.to(self.device).eval()
            log.info(f"{self.modelPath} Loaded to {self.device}")
            return model
        elif self.modelName == 'yolov11':
            # Load the yolov11 model
            model = ultralytics.YOLO(self.modelPath)
            log.info(f"{self.modelPath} Loaded to {self.device}")
            return model
        # elif self.modelName == 'RF-DETR':
        else:
            log.error(f"Model {self.modelName} not supported. Only {list(models.keys())} are supported.")
            sys.exit(1)
        
    def predict(self, image):

       # Dictionary to store valid predictions in consistent format regardless of model
        validPredictions = {}
        validPredictions['Labels'] = []
        validPredictions['Boxes'] = []
        validPredictions['Scores'] = []

        if self.modelName == 'fasterrcnn':
            # Preprocess the image for fasterrcnn
            image = torchvision.transforms.functional.to_tensor(image).to(self.device)
            with torch.no_grad():
                startTime = time.time()
                predictions = self.model([image])[0]
                endTime = time.time()
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if score > self.confidenceThreshold:
                    validPredictions['Labels'].append(CLASSS_LABELS[label-1])
                    validPredictions['Boxes'].append(box)
                    validPredictions['Scores'].append(score)
        if self.modelName == 'yolov11':
            startTime = time.time()
            predictions = self.model(image, verbose=False, show=False, save=False)
            endTime = time.time()

            for prediction in predictions:
                for box, score, label in zip(prediction.boxes.xyxy, prediction.boxes.conf, prediction.boxes.cls):
                    if score > self.confidenceThreshold:
                        validPredictions['Labels'].append(prediction.names[int(label)])
                        validPredictions['Boxes'].append(box.cpu().numpy())
                        validPredictions['Scores'].append(score.cpu().numpy())
        
        inferenceTime = endTime - startTime
        log.info(f"Inference time: {inferenceTime:.4f} seconds")

        return validPredictions, inferenceTime
    
    def benchmark(self, dataPath, warmup=10, runs=100):
        """
        Benchmark the model on a set of images.
        :param dataPath: Path to the directory containing images.
        :param warmup: Number of warmup iterations to run before timing.
        :param runs: Number of runs to average the timing over.
        """
        loader = DataLoader(dataPath)
        log.info(f"Benchmarking {self.modelName} model on {len(loader)} images...")
        outputPath = os.path.join('benchmarks', f"baseline_{self.modelName}.csv")
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        times = []
        numPredictions = []
        # Warm-up
        for i in range(warmup+runs):
            frame = loader[i]
            validPredictions, inferenceTime = self.predict(frame)
            if i >= warmup:
                times.append(inferenceTime)
                numPredictions.append(len(validPredictions['Boxes']))
                log.info(f"Image {i+1}/{len(loader)}: {len(validPredictions['Boxes'])} predictions, inference time: {inferenceTime:.4f} seconds")
        
        # Save results to CSV
        df = pd.DataFrame({
            'Image': [f"Image {i+1}" for i in range(len(times))],
            'Num Predictions': numPredictions,
            'Inference Time (s)': times
        })
        df.to_csv(outputPath, index=False)
        
        avgTime = np.mean(times)
        log.info(f"Average inference time: {avgTime*1000:.2f} ms")
        log.info(f"Average FPS: {1/avgTime:.2f}")
