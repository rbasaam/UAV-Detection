import os
import time
import logging
import numpy as np
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TRTDetector")

# Constants
CLASS_LABELS = ['UAV', 'Airliner', 'Balloon', 'Bird', 'Helicopter']
DEFAULT_ENGINE_DIR = 'engines'

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, name):
        self.host = host_mem
        self.device = device_mem
        self.name = name
    def __str__(self):
        return f"HostDeviceMem(name={self.name}, host_shape={self.host.shape})"
    def __repr__(self):
        return self.__str__()

class TRTDetector:
    """
    Efficient TensorRT detector using execute_async_v3 and explicit tensor bindings.
    """
    def __init__(self, modelName, precision='fp16', confidenceThreshold=0.5):
        # Load engine
        self.modelName = modelName.lower()
        self.precision = precision.lower()
        self.enginePath = os.path.join(DEFAULT_ENGINE_DIR, f"{modelName}_{precision}.trt")
        if not os.path.exists(self.enginePath):
            raise FileNotFoundError(f"Engine file {self.enginePath} not found.")
        self.conf_threshold = confidenceThreshold
        # Logger and runtime
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(self.enginePath, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError("Failed to deserialize engine.")
        log.info(f"Loaded TensorRT engine: {self.enginePath}")
        # Execution context
        self.context = self.engine.create_execution_context()
        # Allocate I/O bindings
        self.inputs, self.outputs, self.bindings, self.stream = self._allocateBuffers()

    def _allocateBuffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        # store names for binding address calls
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            # host & device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            # classify
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem, name))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem, name))
        return inputs, outputs, bindings, stream

    def predict(self, image, show=False, verbose=False):
        # Preprocess
        input_arr = self._preprocess(image)
        inp = self.inputs[0]
        # copy CPU→GPU
        np.copyto(inp.host, input_arr.ravel())
        cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        # bind device addresses
        # must match context.set_tensor_address API: name, ptr
        for io in self.inputs + self.outputs:
            self.context.set_tensor_address(io.name, int(io.device))
        # execute
        start = time.time()
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        inferenceTime = time.time()-start
        # GPU→CPU
        out = self.outputs[0]
        cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()
        # Postprocess
        boxes, scores, classes = self._postprocess(out.host, self.engine.get_tensor_shape(out.name))
        
        validPredictions = {}
        validPredictions['Boxes'] = boxes
        validPredictions['Scores'] = scores
        validPredictions['Labels'] = [CLASS_LABELS[c] for c in classes]

        if verbose:
            log.info(f"Inference: {inferenceTime*1000:.2f} ms")
            log.info(f"Boxes:\n{boxes}")
            log.info(f"Scores:\n{scores}")
            log.info(f"Classes:\n{classes}")
        if show:
            self._showResults(image, boxes, scores, classes)


        return validPredictions, inferenceTime

    def _preprocess(self, image):
        # assume single input
        inp = self.inputs[0]
        shape = tuple(self.engine.get_tensor_shape(inp.name))  # e.g. (1,3,H,W)
        _, C, H, W = shape
        resized = cv2.resize(image, (W, H)).astype(np.float32) / 255.0
        # HWC→CHW
        transposed = np.transpose(resized, (2,0,1))
        # add batch dim
        batched = np.expand_dims(transposed, axis=0)
        return np.ascontiguousarray(batched)

    def _postprocess(self, flat_out, shape):
        # shape is tuple e.g. (1,N,6)
        total = int(np.prod(shape))
        arr = np.array(flat_out[:total])  # ensure correct length
        arr = arr.reshape(shape)
        arr = arr[0]  # remove batch dim
        boxes = arr[:, :4]
        scores = arr[:, 4]
        classes = arr[:, 5].astype(int)
        mask = scores > self.conf_threshold
        return boxes[mask], scores[mask], classes[mask]

    def _showResults(self, image, boxes, scores, classes):
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for box, score, cls in zip(boxes, scores, classes):
            x1,y1,x2,y2 = box
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{CLASS_LABELS[cls]}:{score:.2f}",
                    color='white', fontsize=12,
                    bbox=dict(facecolor='red', alpha=0.5))
        plt.show()
        log.info("Displayed results.")
        return boxes, scores, classes
    
    def benchmark(self, loader, warmup=10, runs=100):
        log.info(f"Benchmarking {self.modelName} TensorRT Engine with {self.precision} precision...")

        times = []
        numPredictions = []

        for i in range(warmup+runs):
            frame = loader[i % len(loader)]
            validPredictions, inferenceTime = self.predict(frame)
            if i >= warmup:
                times.append(inferenceTime)
                numPredictions.append(len(validPredictions['Boxes']))
                log.info(f"Image {i-warmup+1}/{runs}: {len(validPredictions['Boxes'])} predictions, "
                         f"inference time: {inferenceTime*1000:.4f} ms")
        
        avgTime = np.mean(times)
        log.info(f"Average Inference Time: {avgTime*1000:.2f} ms over {runs} runs")
        log.info(f"Average FPS: {1/avgTime:.2f}")

        df = pd.DataFrame({
            'Image': [f"Image {i+1}" for i in range(runs)],
            'Num Predictions': numPredictions,
            'Inference Time (s)': times
        })
        df.to_csv(f"benchmarks/{self.modelName}_{self.precision}_benchmark.csv", index=False)
        log.info(f"Benchmark results saved to {self.modelName}_{self.precision}_benchmark.csv")

