from model_loader import Detector
import cv2
from data_loader import DataLoader

imageDir = 'data/images/DJI_0002/'
loader = DataLoader(imageDir)

fasterRCNN = Detector(modelName = 'FasterRCNN', confidenceThreshold = 0.7)
fasterRCNN.predict(loader[100], verbose=True, plot=False)
fasterRCNN.exportONNX()

yolov11 = Detector(modelName = 'YOLOv11', confidenceThreshold = 0.7)
yolov11.predict(loader[100], verbose=True, plot=False)
yolov11.exportONNX()


