from detector import Detector
from loader import DataLoader

DATA_PATH = 'data/images/DJI_0002/'
imgIdx = 10

loader = DataLoader(DATA_PATH)
# loader.preview(imgIdx)
loader.getImageInfo(imgIdx)
detector = Detector(modelName='yolov11', confidenceThreshold=0.7)
detector.predict(loader[imgIdx], show=True, verbose=True)