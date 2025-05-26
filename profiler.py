from detector import Detector
from trt_detector import TRTDetector
from loader import DataLoader
import time

DATA_PATH = 'data/images/DJI_0002/'

loader = DataLoader(DATA_PATH)

MODEL = 'yolov11'
CONFIDENCE_THRESHOLD = 0.7

yolo_base = Detector(modelName=MODEL, confidenceThreshold=CONFIDENCE_THRESHOLD)
yolo_fp16 = TRTDetector(modelName=MODEL, precision='fp16', confidenceThreshold=CONFIDENCE_THRESHOLD)
yolo_fp32 = TRTDetector(modelName=MODEL, precision='fp32', confidenceThreshold=CONFIDENCE_THRESHOLD)

for model in [yolo_base, yolo_fp16, yolo_fp32]:
    model.benchmark(loader, warmup=20, runs=200)
    


