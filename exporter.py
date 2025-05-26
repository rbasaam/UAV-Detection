"""
Model exporter module for UAV Detection.

This module provides functionality for exporting trained models to ONNX format
to enable deployment across different platforms and inference engines.
"""

import os
import sys
import torch
import logging
import argparse
import torch.onnx
from detector import Detector

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Exporter")

# Constants
SUPPORTED_MODELS = ['fasterrcnn', 'yolov11', 'rfdetr']
DEFAULT_EXPORT_DIR = 'models'


class Exporter:
    """
    Model exporter class for converting PyTorch models to ONNX format.
    
    Supports exporting multiple detection model architectures including
    Faster R-CNN, YOLOv11, and RFDETR.
    """
    
    def __init__(self, modelName, confidenceThreshold=0.5):
        """
        Initialize the exporter with a specified model.
        
        Args:
            modelName (str): Name of the model to export ('fasterrcnn', 'yolov11', or 'rfdetr')
            confidenceThreshold (float): Confidence threshold for detection (0.0-1.0)
        """
        self.modelName = modelName.lower()
        self.confidenceThreshold = confidenceThreshold
        
        # Validate model name
        if self.modelName not in SUPPORTED_MODELS:
            log.error(f"Model {self.modelName} not supported. Only {', '.join(SUPPORTED_MODELS)} are supported.")
            sys.exit(1)
            
        log.info(f"Preparing to export {self.modelName} model to ONNX format...")
        
        # Initialize detector to load the model
        self.detector = Detector(modelName=self.modelName, confidenceThreshold=confidenceThreshold)
        self.model = self.detector.model
        
        # Set device from the detector
        self.device = self.detector.device
        log.info(f"Using device: {self.device}")
        
        # Define the export path
        os.makedirs(DEFAULT_EXPORT_DIR, exist_ok=True)
        self.exportPath = os.path.join(DEFAULT_EXPORT_DIR, f"{self.modelName}.onnx")
        self.exported = False
    
    def exportONNX(self):
        """
        Export the model to ONNX format.
        
        Each model type requires slightly different export parameters and handling.
        
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Export model based on its type
            if self.modelName == 'fasterrcnn':
                self._exportFasterRCNN()
            elif self.modelName == 'yolov11':
                self._exportYOLO()
            elif self.modelName == 'rfdetr':
                self._exportRFDETR()
                
            return self.exported
            
        except Exception as e:
            log.error(f"Export failed: {str(e)}")
            return False
            
    def _exportFasterRCNN(self):
        """Export Faster R-CNN model to ONNX."""
        log.info("Exporting Faster R-CNN model...")
        
        # Create dummy input for tracing
        dummy_input = torch.randn(1, 3, 1530, 2720).to(self.device)

        # Export using ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            self.exportPath,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['boxes', 'labels', 'scores'],
        )
        
        log.info(f"Exported Faster R-CNN model to {self.exportPath}")
        self.exported = True
        
    def _exportYOLO(self):
        """Export YOLOv11 model to ONNX."""
        log.info("Exporting YOLOv11 model...")
        
        # For YOLO models, use the built-in export method
        self.model.export(
            format='onnx',
            imgsz=(1530, 2720),     # Input size
            opset=17,
            nms=True,
            simplify=True,
            half=True,              # Use half precision if available
            device=0,               # Use GPU if available
        )
        
        log.info(f"Exported YOLOv11 model to {self.exportPath}")
        self.exported = True
        
    def _exportRFDETR(self):
        """Export RFDETR transformer model to ONNX."""
        log.info("Exporting RFDETR model...")
        
        self.model.export()
        
        log.info(f"Exported RFDETR model to {self.exportPath}")
        self.exported = True


def main():
    """
    Entry point when script is run directly.
    
    Parses command line arguments and exports the specified model.
    """
    parser = argparse.ArgumentParser(
        description="Export trained detection models to ONNX format for deployment."
    )
    
    parser.add_argument(
        '-m', '--model', 
        type=str, 
        required=True, 
        help="Model name for export", 
        choices=SUPPORTED_MODELS
    )
    
    parser.add_argument(
        '-c', '--confidence', 
        type=float, 
        default=0.5,
        help="Confidence threshold for detection (0.0-1.0)"
    )
    
    args = parser.parse_args()

    # Export the model
    exporter = Exporter(args.model, args.confidence)
    success = exporter.exportONNX()
    
    if success:
        log.info("Export completed successfully.")
        sys.exit(0)
    else:
        log.error("Export failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
