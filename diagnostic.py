from detector import Detector
from trt_detector import TRTDetector
from loader import DataLoader

DATA_PATH = 'data/images/DJI_0002/'
imgIdx = 10

loader = DataLoader(DATA_PATH)
# loader.preview(imgIdx)
loader.getImageInfo(imgIdx)

rfdetr_base = Detector(modelName='rfdetr', confidenceThreshold=0.7)
rfdetr_base.predict(loader[imgIdx], show=True, verbose=True)
rfdetr_fp16 = TRTDetector(modelName='rfdetr', precision='fp16', confidenceThreshold=0.7)
rfdetr_fp16.predict(loader[imgIdx], show=True, verbose=True)
