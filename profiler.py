from detector import Detector

DATA_PATH = 'data/images/DJI_0002'
MODELS = ['fasterrcnn', 'yolov11']

detector = Detector(modelName='yolov11')
detector.benchmark(
    dataPath=DATA_PATH,
    warmup=10,
    runs=300
)
