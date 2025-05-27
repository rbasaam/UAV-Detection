import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detector import Detector
from trt_detector import TRTDetector
from loader import DataLoader

DATA_PATH = 'data/images/DJI_0002/'

loader = DataLoader(DATA_PATH)

MODEL = 'rfdetr'
CONFIDENCE_THRESHOLD = 0.7
WARMUP = 20
NUMRUNS = 300

# yolo_base = Detector(modelName='yolov11', confidenceThreshold=CONFIDENCE_THRESHOLD)
yolo_fp16 = TRTDetector(modelName='yolov11', precision='fp16', confidenceThreshold=CONFIDENCE_THRESHOLD)
# yolo_fp32 = TRTDetector(modelName='yolov11', precision='fp32', confidenceThreshold=CONFIDENCE_THRESHOLD)

yolo_fp16.predict(loader[15], show=True, verbose=True)


# plt.figure(figsize=(12, 6))
# benchmarks = {}
# for model in [yolo_base, yolo_fp16, yolo_fp32]:
#     benchmarks[model.modelName] = model.benchmark(
#         loader=loader,
#         warmup=WARMUP,
#         runs=NUMRUNS,
#     )
#     plt.plot(np.arange(NUMRUNS), benchmarks[model.modelName]['Inference Time (ms)'], label=f"{model.modelName} ({model.precision if hasattr(model, 'precision') else 'base'})")
# plt.xlabel('Run Number')
# plt.ylabel('Inference Time (ms)')
# plt.title('Inference Time Comparison for YOLOv11 Models')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('benchmarks/yolov11_inference_comparison.png')
# plt.show()

rfdetr_fp16 = TRTDetector(modelName='rfdetr', precision='fp16', confidenceThreshold=CONFIDENCE_THRESHOLD)
# rfdetr_fp32 = TRTDetector(modelName='rfdetr', precision='fp32', confidenceThreshold=CONFIDENCE_THRESHOLD)

rfdetr_fp16.predict(loader[15], show=True, verbose=True)





