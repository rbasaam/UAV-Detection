import torch
import torch.onnx
import torchvision
import os
import time
import pandas as pd
import ultralytics
import matplotlib.pyplot as plt
import matplotlib.patches as patches


CLASSS_LABELS = ['UAV', 'Airliner', 'Balloon', 'Bird', 'Helicopter']
models = {
    'FasterRCNN': 'fasterrcnn.pth',
    'YOLOv11': 'yolov11.pt',
    'RF-DETR': 'transformer.pth',
    'FasterRCNN-ONNX': 'FasterRCNN.onnx'
}

class Detector():
    def __init__ (self, modelName, confidenceThreshold=0.6):
        self.modelName = modelName
        if modelName not in models:
            raise ValueError(f"Model {modelName} not supported. Only {list(models.keys())} are supported.")
        self.modelPath = os.path.join('models', models[modelName])
        self.confidenceThreshold = confidenceThreshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.loadModel()
    
    def loadModel(self):
        if not os.path.exists(self.modelPath):
            raise FileNotFoundError(f"Model file {self.modelPath} not found.")
        if self.modelName == 'FasterRCNN':
            # Load the FasterRCNN mode
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None, num_classes=len(CLASSS_LABELS)+1
            )
            checkpoint = torch.load(self.modelPath, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.to(self.device).eval()
            return model
        elif self.modelName == 'YOLOv11':
            # Load the YOLOv11 model
            model = ultralytics.YOLO(self.modelPath)
            return model
        # elif self.modelName == 'RF-DETR':
        else:
            raise ValueError(f"Model {self.modelName} not supported. Only {models.keys()} are supported.")
        
    def predict(self, image, verbose=False, plot=False):

       # Dictionary to store valid predictions in consistent format regardless of model
        validPredictions = {}
        validPredictions['Labels'] = []
        validPredictions['Boxes'] = []
        validPredictions['Scores'] = []

        if self.modelName == 'FasterRCNN':
            # Preprocess the image for FasterRCNN
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
        if self.modelName == 'YOLOv11':
            startTime = time.time()
            predictions = self.model(image, verbose=False, show=False, save=False)
            endTime = time.time()

            for prediction in predictions:
                for box, score, label in zip(prediction.boxes.xyxy, prediction.boxes.conf, prediction.boxes.cls):
                    if score > self.confidenceThreshold:
                        validPredictions['Labels'].append(prediction.names[int(label)])
                        validPredictions['Boxes'].append(box.cpu().numpy())
                        validPredictions['Scores'].append(score.cpu().numpy())

        if verbose:
            print('-'*60)
            print(f"Model: {self.modelName}")
            print(f"Model Path: {self.modelPath}")
            print(f"Image Shape: {image.shape}")
            print(f"Inference Time: {endTime - startTime:.2f} seconds")
            if len(validPredictions['Labels'])>0:
                print(f"{len(validPredictions['Labels'])} valid predictions")
                print(pd.DataFrame(validPredictions).to_string(index=False))
            else:
                print(f"No predictions made with scores > {self.confidenceThreshold}")

        if plot:
            fig, ax = plt.subplots(figsize=(12,8))
            if self.modelName == 'FasterRCNN':
                image = image.cpu().numpy().transpose(1, 2, 0)
            ax.imshow(image)
            for box, label, score in zip(validPredictions['Boxes'], validPredictions['Labels'], validPredictions['Scores']):
                x1, y1, x2, y2 = box
                width, height = x2-x1, y2-y1
                ax.add_patch(
                    patches.Rectangle(
                        xy=(x1, y1),
                        width=width,
                        height=height,
                        linewidth = 2,
                        edgecolor = 'green',
                        facecolor = 'none'
                    )
                )

                ax.text(
                    x = x1,
                    y = y1 - 10,
                    s = f"{label}: {score:.2f}",
                    color = 'black',
                    bbox = dict(facecolor='black', alpha=0.5, pad=3)
                )

            plt.axis('off')
            plt.title(f"{self.modelName} Predictions", fontsize=16)
            plt.tight_layout()
            plt.show()
    
    def exportONNX(self):
        outputPath = os.path.join('models', f"{self.modelName.lower()}.onnx")
        if self.modelName == 'FasterRCNN':
            dummyInput = torch.randn(1, 3, 1530, 2720).to(self.device)
            self.model.eval()
            torch.onnx.export(
                self.model,
                dummyInput,
                outputPath,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['boxes', 'labels', 'scores'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'boxes': {0: 'numDetections'},
                    'labels': {0: 'numDetections'},
                    'scores': {0: 'numDetections'}
                },
            )
        elif self.modelName == 'YOLOv11':
            self.model.export(format='onnx')
        print(f"Model exported to {outputPath}")
        

