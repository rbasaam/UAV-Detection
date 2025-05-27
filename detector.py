"""
UAV Detection module using multiple object detection models.
This module provides a unified interface for different object detection models.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import torchvision
import ultralytics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rfdetr import RFDETRBase

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Detector")

# Constants
CLASS_LABELS = ['UAV', 'Airliner', 'Balloon', 'Bird', 'Helicopter']
MODEL_PATHS = {
    'fasterrcnn': 'fasterrcnn.pth',
    'yolov11': 'yolov11.pt',
    'rfdetr': 'rfdetr.pth',
}


class Detector:
    """
    Unified detector class supporting multiple object detection models.
    
    Currently supported models:
    - Faster R-CNN (PyTorch)
    - YOLOv11 (Ultralytics)
    - RFDETR (Transformer-based)
    """
    
    def __init__(self, modelName, confidenceThreshold=0.6):
        """
        Initialize the detector with the specified model.
        
        Args:
            modelName (str): Name of the model to use ('fasterrcnn', 'yolov11', or 'rfdetr')
            confidenceThreshold (float): Detection confidence threshold (0.0-1.0)
        """
        self.modelName = modelName
        self.confidenceThreshold = confidenceThreshold
        self.modelPath = os.path.join('models', MODEL_PATHS[modelName])
        
        # Verify model file exists
        if not os.path.exists(self.modelPath):
            log.error(f"Model file {self.modelPath} not found.")
            sys.exit(1)
        log.info(f"Model file found: {self.modelPath}")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self.loadModel()
           
    def loadModel(self):
        """
        Load the specified model.
        
        Returns:
            model: The loaded detection model
        """
        if self.modelName == 'fasterrcnn':
            return self._loadFasterRCNN()
        elif self.modelName == 'yolov11':
            return self._loadYOLO()
        elif self.modelName == 'rfdetr':
            return self._loadRFDETR()
    
    def _loadFasterRCNN(self):
        """Load and configure Faster R-CNN model."""
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=None, 
            weights_backbone=None, 
            num_classes=len(CLASS_LABELS)+1
        )
        checkpoint = torch.load(self.modelPath, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device).eval()
        log.info(f"{self.modelPath} Loaded to {self.device}")
        return model
    
    def _loadYOLO(self):
        """Load YOLOv11 model."""
        model = ultralytics.YOLO(self.modelPath)
        log.info(f"{self.modelPath} Loaded to {self.device}")
        return model
    
    def _loadRFDETR(self):
        """Load RFDETR transformer model."""
        os.environ["HF_TOKEN"] = "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"
        os.environ["ROBOFLOW_API_KEY"] = "dC56SwwV9TulAW12Tkj6"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        model = RFDETRBase(pretrain_weights=self.modelPath)
        return model
        
    def predict(self, image, show=False, verbose=False):
        """
        Run object detection on an image.
        
        Args:
            image: Input image (numpy array or tensor)
            show (bool): Whether to display the detection results
            verbose (bool): Whether to print detailed information
            
        Returns:
            tuple: (validPredictions, inferenceTime)
                - validPredictions (dict): Dictionary with detection results
                - inferenceTime (float): Inference time in seconds
        """
        # Dictionary to store valid predictions in consistent format
        validPredictions = {
            'Labels': [],
            'Boxes': [],
            'Scores': []
        }

        # Run model-specific prediction
        if self.modelName == 'fasterrcnn':
            validPredictions, startTime, endTime = self._predictFasterRCNN(image, validPredictions)
        elif self.modelName == 'yolov11':
            validPredictions, startTime, endTime = self._predictYOLO(image, validPredictions)
        elif self.modelName == 'rfdetr':
            validPredictions, startTime, endTime = self._predictRFDETR(image, validPredictions)
        
        # Calculate inference time
        inferenceTime = endTime - startTime
        
        # Display verbose information if requested
        if verbose:
            self._displayVerboseInfo(validPredictions, inferenceTime)
            
        # Show results if requested
        if show:
            self._visualizeResults(image, validPredictions)
            
        return validPredictions, inferenceTime
    
    def _predictFasterRCNN(self, image, validPredictions):
        """Run prediction with Faster R-CNN model."""
        # Preprocess the image for fasterrcnn
        image_tensor = torchvision.transforms.functional.to_tensor(image).to(self.device)
        with torch.no_grad():
            startTime = time.time()
            predictions = self.model([image_tensor])[0]
            endTime = time.time()
            
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score > self.confidenceThreshold:
                validPredictions['Labels'].append(CLASS_LABELS[label-1])
                validPredictions['Boxes'].append(box)
                validPredictions['Scores'].append(score)
                
        return validPredictions, startTime, endTime
    
    def _predictYOLO(self, image, validPredictions):
        """Run prediction with YOLOv11 model."""
        startTime = time.time()
        predictions = self.model(image, verbose=False, show=False, save=False)
        endTime = time.time()

        for prediction in predictions:
            for box, score, label in zip(prediction.boxes.xyxy, prediction.boxes.conf, prediction.boxes.cls):
                if score > self.confidenceThreshold:
                    validPredictions['Labels'].append(prediction.names[int(label)])
                    validPredictions['Boxes'].append(box.cpu().numpy())
                    validPredictions['Scores'].append(score.cpu().numpy())
        
        return validPredictions, startTime, endTime
    
    def _predictRFDETR(self, image, validPredictions):
        """Run prediction with RFDETR model."""
        startTime = time.time()
        predictions = self.model.predict(image, threshold=self.confidenceThreshold)
        endTime = time.time()
        
        if predictions is not None:
            for label, box, score in zip(predictions.class_id, predictions.xyxy, predictions.confidence):
                if score > self.confidenceThreshold:
                    validPredictions['Labels'].append(CLASS_LABELS[label])
                    validPredictions['Boxes'].append(box)
                    validPredictions['Scores'].append(score)
        
        return validPredictions, startTime, endTime
    
    def _displayVerboseInfo(self, validPredictions, inferenceTime):
        """Display detailed information about predictions."""
        log.info(f"Model: {self.modelName}, Inference Time: {inferenceTime:.4f} seconds")
        log.info(f"Predictions: {len(validPredictions['Boxes'])} boxes detected with scores above threshold {self.confidenceThreshold}")
        print('-'*50)
        
        for i, (label, box, score) in enumerate(zip(
                validPredictions['Labels'],
                validPredictions['Boxes'],
                validPredictions['Scores'])):
            print(f"Prediction {i+1}: {label} ({score:.2f})")
            print(f"Box coordinates: {box}")
    
    def _visualizeResults(self, image, validPredictions):
        """Visualize detection results on the image."""
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.set_title(f"{self.modelName} Base Detector", fontsize=16)
        # Handle different image formats
        if isinstance(image, np.ndarray):
            # If image is numpy array in RGB format
            ax.imshow(image)
        elif isinstance(image, torch.Tensor):
            # If image is a PyTorch tensor
            img_np = image.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_np)
        
        # Draw bounding boxes and labels
        for label, box, score in zip(
                validPredictions['Labels'],
                validPredictions['Boxes'],
                validPredictions['Scores']):
            
            # Extract coordinates
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            
            # Create and add rectangle
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with score
            ax.text(
                xmin-15, ymin-30,
                f'{label} ({score:.2f})',
                fontsize=12, color='white',
                bbox=dict(facecolor='green', alpha=0.5)
            )
        
        plt.axis('off')
        plt.show()
    
    def benchmark(self, loader, warmup=10, runs=100):
        """
        Benchmark the model on a set of images.
        
        Args:
            dataPath (str): Path to the directory containing images
            warmup (int): Number of warmup iterations before timing
            runs (int): Number of runs to average timing over
        """
        # Load dataset
        log.info(f"Benchmarking {self.modelName} model on {len(loader)} images...")
        
        # Prepare output path
        outputPath = os.path.join('benchmarks', f"{self.modelName}_baseline.csv")
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        
        times = []
        numPredictions = []
        
        # Run benchmark (warmup + timed runs)
        for i in range(warmup + runs):
            frame = loader[i % len(loader)]  # Loop through images if needed
            validPredictions, inferenceTime = self.predict(frame)
            
            # Only collect stats after warmup
            if i >= warmup:
                times.append(inferenceTime)
                numPredictions.append(len(validPredictions['Boxes']))
                log.info(f"Image {i-warmup+1}/{runs}: {len(validPredictions['Boxes'])} predictions, "
                         f"inference time: {inferenceTime*1000:.4f} ms")
        
        # Save results to CSV
        df = pd.DataFrame({
            'Image': [f"Image {i+1}" for i in range(len(times))],
            'Num Predictions': numPredictions,
            'Inference Time (ms)': [t*1000 for t in times]  # Convert to milliseconds
        })
        df.to_csv(outputPath, index=False)
        
        # Calculate and report statistics
        avgTime = np.mean(times)
        log.info(f"Average inference time: {avgTime*1000:.2f} ms")
        log.info(f"Average FPS: {1/avgTime:.2f}")
        log.info(f"Benchmark results saved to {outputPath}")
        return df
