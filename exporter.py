import os
import sys
import torch
import logging
import argparse
import torch.onnx
import torchvision
import ultralytics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Exporter")

models = {
    'fasterrcnn': 'fasterrcnn.pth',
    'yolov11': 'yolov11.pt',
}

class Exporter:
    def __init__(self, modelName):
        self.modelName = modelName.lower()
        if self.modelName not in ['fasterrcnn', 'yolov11']:
            log.error(f"Model {self.modelName} not supported. Only {list(models.keys())} are supported.")
            sys.exit(1)
        log.info(f"Exporting {self.modelName} model to ONNX format...")
        self.modelPath = os.path.join('models', models[self.modelName])
        if not os.path.exists(self.modelPath):
            log.error(f"Model file {self.modelPath} not found.")
            sys.exit(1)
        log.info(f"Model file found: {self.modelPath}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device: {self.device}")
        self.model = self.loadModel()
        self.exportPath = os.path.join('models', f"{self.modelName}.onnx")
        self.exported = False
    
    def loadModel(self):
        if self.modelName == 'fasterrcnn':
            # Load the FasterRCNN model
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None, num_classes=6
            )
            checkpoint = torch.load(self.modelPath, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.to(self.device).eval()
            log.info(f"Model loaded: {self.modelPath}")
            return model
        elif self.modelName == 'yolov11':
            model = ultralytics.YOLO(self.modelPath)
            log.info(f"Model loaded: {self.modelPath}")
            return model
    
    def exportONNX(self):
        if self.modelName == 'fasterrcnn':
            dummy_input = torch.randn(1, 3, 1530, 2720).to(self.device)

            torch.onnx.export(
                self.model,
                dummy_input,
                self.exportPath,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['boxes', 'labels', 'scores'],
                dynamic_axes={
                    'input': {2: 'height', 3: 'width'},
                    # 'boxes': {0: 'detections'},
                    # 'labels': {0: 'detections'},
                    # 'scores': {0: 'detections'},
                    },
                # dynamo=True
            )
            log.info(f"Exported FasterRCNN model to {self.exportPath}")
            self.exported = True
        elif self.modelName == 'yolov11':
            self.model.export(format='onnx')
            log.info(f"Exported YOLOv11 model to {self.exportPath}")
            self.exported = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX format.")
    parser.add_argument('--model', type=str, required=True, help="Model name: 'fasterrcnn' or 'yolov11'")
    args = parser.parse_args()

    exporter = Exporter(args.model)
    exporter.exportONNX()


        
        