import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def buildEngine(onnxPath, enginePath):
    """
    Build a TensorRT engine from an ONNX model and save it to a file.
    """
    