import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detector import Detector
from trt_detector import TRTDetector
from loader import DataLoader

DATA_PATH = 'data/images/DJI_0002/'
LOADER = DataLoader(DATA_PATH)
MODEL = 'rfdetr'
CONFIDENCE_THRESHOLD = 0.7
WARMUP = 20
NUMRUNS = 300

def profileResults(modelName, loader, confidenceThreshold, warmup=10, runs=100):
    baseModel = Detector(modelName=modelName, confidenceThreshold=confidenceThreshold)
    fp16Model = TRTDetector(modelName=modelName, precision='fp16', confidenceThreshold=confidenceThreshold)
    fp32Model = TRTDetector(modelName=modelName, precision='fp32', confidenceThreshold=confidenceThreshold)

    plt.figure(figsize=(12, 6))
    plt.xlabel('Run Number')
    plt.ylabel('Inference Time (ms)')

    benchmarks = {}
    for model in [baseModel, fp16Model, fp32Model]:
        benchmarks[model.modelName] = model.benchmark(
            loader=loader,
            warmup=warmup,
            runs=runs,
        )
        plt.plot(np.arange(runs), benchmarks[model.modelName]['Inference Time (ms)'], label=f"{model.modelName} ({model.precision if hasattr(model, 'precision') else 'base'})")

    plt.title(f'Inference Time Comparison for {modelName} Models')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'benchmarks/{modelName}_inference_comparison.png')
    plt.show()
    plt.close()

    

profileResults(MODEL, LOADER, CONFIDENCE_THRESHOLD, WARMUP, NUMRUNS)





