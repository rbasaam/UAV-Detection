from detector import Detector
from trt_detector import TRTDetector
from loader import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DATA_PATH = 'data/images/DJI_0002/'
# imgIdx = 10

# loader = DataLoader(DATA_PATH)
# # loader.preview(imgIdx)
# loader.getImageInfo(imgIdx)

# rfdetr_base = Detector(modelName='rfdetr', confidenceThreshold=0.7)
# rfdetr_base.predict(loader[imgIdx], show=True, verbose=True)
# rfdetr_fp16 = TRTDetector(modelName='rfdetr', precision='fp16', confidenceThreshold=0.7)
# rfdetr_fp16.predict(loader[imgIdx], show=True, verbose=True)

def benchmarkPlot(modelName):

    baseFile = f'benchmarks/{modelName}_baseline.csv'
    fp16File = f'benchmarks/{modelName}_fp16_benchmark.csv'
    fp32File = f'benchmarks/{modelName}_fp32_benchmark.csv'

    baseDf = pd.read_csv(baseFile)
    fp16Df = pd.read_csv(fp16File)
    fp32Df = pd.read_csv(fp32File)

    plt.figure(figsize=(12, 6))
    plt.xlabel('Run Number')
    plt.ylabel('Inference Time (ms)')

    plt.plot(np.arange(len(baseDf)), baseDf['Inference Time (ms)'], label=f'{modelName} (Base) | Avg Inference: {baseDf["Inference Time (ms)"].mean():.2f} ms | Avg FPS: {1000/baseDf["Inference Time (ms)"].mean():.2f}', color='blue', linestyle='-')
    plt.axhline(y=baseDf['Inference Time (ms)'].mean(), color='blue', linestyle='--')

    plt.plot(np.arange(len(fp16Df)), fp16Df['Inference Time (ms)'], label=f'{modelName} (FP16) | Avg Inference: {fp16Df["Inference Time (ms)"].mean():.2f} ms | Avg FPS: {1000/fp16Df["Inference Time (ms)"].mean():.2f}', color='orange', linestyle='-')
    plt.axhline(y=fp16Df['Inference Time (ms)'].mean(), color='orange', linestyle='--')

    plt.plot(np.arange(len(fp32Df)), fp32Df['Inference Time (ms)'], label=f'{modelName} (FP32) | Avg Inference: {fp32Df["Inference Time (ms)"].mean():.2f} ms | Avg FPS: {1000/fp32Df["Inference Time (ms)"].mean():.2f}', color='green', linestyle='-')
    plt.axhline(y=fp32Df['Inference Time (ms)'].mean(), color='green', linestyle='--')

    plt.title(f'Inference Time Comparison for {modelName} Models')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'benchmarks/{modelName}_inference_comparison.png')
    plt.show()
    plt.close()

benchmarkPlot('rfdetr')
benchmarkPlot('yolov11')



