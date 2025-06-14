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
import gc

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
        
        # Create reusable CUDA events for timing
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()
        
        self._is_initialized = True
        log.info(f"Initialized TRTDetector for {self.modelName} with {self.precision} precision.")
    
    def __del__(self):
        """Destructor to ensure resources are properly released."""
        self.cleanup()

    def cleanup(self):
        """Release all allocated CUDA resources."""
        if hasattr(self, '_is_initialized') and self._is_initialized:
            log.info(f"Cleaning up TensorRT resources for {self.modelName}")
            
            # Free CUDA memory
            if hasattr(self, 'inputs'):
                for inp in self.inputs:
                    if hasattr(inp, 'device') and inp.device:
                        try:
                            inp.device.free()
                            inp.device = None
                        except Exception as e:
                            log.error(f"Failed to free device memory for {inp.name}: {e}")
            
            if hasattr(self, 'outputs'):
                for out in self.outputs:
                    if hasattr(out, 'device') and out.device:
                        try:
                            out.device.free()
                            out.device = None
                        except Exception as e:
                            log.error(f"Failed to free device memory for {out.name}: {e}")
            
            # Release CUDA stream - Note: removing this as it causes issues with pyCUDA's context management
            # if hasattr(self, 'stream'):
            #     self.stream.synchronize()
            #     del self.stream
            
            # Release TensorRT resources
            if hasattr(self, 'context'):
                del self.context
            
            if hasattr(self, 'engine'):
                del self.engine
            
            if hasattr(self, 'runtime'):
                del self.runtime
            
            # Mark as cleaned up
            self._is_initialized = False
            
            # Force garbage collection
            gc.collect()
            
            # Empty CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log.info("Cleared CUDA cache.")
            
            log.info("TensorRT resources cleaned up successfully.")

    def _allocateBuffers(self):
        """Allocate all necessary buffers for inputs and outputs."""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        # Allocate memory for all input/output tensors
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = tuple(self.engine.get_tensor_shape(name))
            
            # Handle dynamic dimensions (negative values)
            if -1 in shape:
                # Replace negative dimensions with defaults
                shape = tuple([max(1, d) for d in shape])
                log.info(f"Tensor {name} has dynamic shape, using {shape}")
            
            # Calculate buffer size
            size = int(np.prod(shape))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Create host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            # Store in appropriate list
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem, name))
                log.info(f"Allocated input buffer: {name}, shape: {shape}, size: {size}")
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem, name))
                log.info(f"Allocated output buffer: {name}, shape: {shape}, size: {size}")
                
        return inputs, outputs, bindings, stream

    def predict(self, image, show=False, verbose=False):
        """Run inference on an image and return detection results."""
        # Preprocess
        input_arr = self._preprocess(image)
        inp = self.inputs[0]
        
        # Copy CPU→GPU
        np.copyto(inp.host, input_arr.ravel())
        cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        # Bind device addresses
        for io in self.inputs + self.outputs:
            self.context.set_tensor_address(io.name, int(io.device))
        
        # Execute with timing
        self.start_event.record(self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.end_event.record(self.stream)
        
        # Synchronize stream and calculate time
        self.stream.synchronize()
        inferenceTime = self.start_event.time_till(self.end_event) * 1e-3  # ms to seconds
        
        # GPU→CPU for all outputs
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()
        
        # Model-specific postprocessing
        if self.modelName == 'rfdetr':
            # For RFDETR with multiple outputs
            boxes, scores, classes = self._postprocess_rfdetr()
        else:
            # For YOLO with single output
            out = self.outputs[0]
            boxes, scores, classes = self._postprocess_yolo(out.host, self.engine.get_tensor_shape(out.name))
        
        validPredictions = {}
        validPredictions['Boxes'] = boxes
        validPredictions['Scores'] = scores
        validPredictions['Labels'] = [CLASS_LABELS[c] for c in classes]

        if verbose:
            log.info(f"Model: {self.modelName}, Inference Time: {inferenceTime:.4f} seconds")
            log.info(f"Predictions: {len(validPredictions['Boxes'])} boxes detected with scores above threshold {self.conf_threshold}")
            print('-'*50)
            for i, (label, box, score) in enumerate(zip(
                    validPredictions['Labels'],
                    validPredictions['Boxes'],
                    validPredictions['Scores'])):
                print(f"Prediction {i+1}: {label} ({score:.2f})")
                print(f"Box coordinates: {box}")
            print('-'*50)
        
        if show:
            self._visualizeResults(image, validPredictions)

        return validPredictions, inferenceTime

    def _postprocess_rfdetr(self):
        # Get raw outputs
        boxes_out, logits_out = self.outputs
        boxes_shape  = self.engine.get_tensor_shape(boxes_out.name)
        logits_shape = self.engine.get_tensor_shape(logits_out.name)

        # Reshape and strip batch dim
        boxes  = np.reshape(boxes_out.host,  boxes_shape)[0]   # (N,4)
        logits = np.reshape(logits_out.host, logits_shape)[0]  # (N,5)

        # ---- SOFTMAX STEP ----
        # Convert logits to probabilities in a numerically stable way
        m    = np.max(logits, axis=1, keepdims=True)             # (N,1)
        exp  = np.exp(logits - m)                                # (N,5)
        probs = exp / np.sum(exp, axis=1, keepdims=True)         # (N,5)

        # Pick class by argmax on logits, and its probability as the score
        classes = np.argmax(logits, axis=1)                      # (N,)
        scores  = probs[np.arange(probs.shape[0]), classes]      # (N,)

        # Filter by confidence threshold
        mask    = scores > self.conf_threshold
        boxes   = boxes[mask]
        scores  = scores[mask]
        classes = classes[mask]

        # Scale normalized boxes back to image size, if needed
        if boxes.size and boxes.max() <= 1.0:
            boxes[:, [0,2]] *= self.orig_width
            boxes[:, [1,3]] *= self.orig_height

        # Convert from (xc, yc, w, h) to (x1, y1, x2, y2)
        xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes = np.column_stack((xc - w/2, yc - h/2, xc + w/2, yc + h/2))

        return boxes, scores, classes

    def _postprocess_yolo(self, flat_out, shape):
        """Process YOLO output tensor with format [batch, num_detections, 6]."""
        # Handle dynamic dimensions in shape
        pos_shape = []
        for dim in shape:
            if dim < 0:
                pos_shape.append(300)  # Default for dynamic dimension (max detections)
            else:
                pos_shape.append(dim)
        
        # Calculate total size and reshape
        total = int(np.prod(pos_shape))
        arr = np.array(flat_out[:total])  # ensure correct length
        arr = arr.reshape(pos_shape)
        arr = arr[0]  # remove batch dimension
        
        # Extract detection data
        boxes = arr[:, :4]
        scores = arr[:, 4]
        classes = arr[:, 5].astype(int)
        
        # Handle out-of-range class IDs
        if len(classes) > 0 and np.any(classes >= len(CLASS_LABELS)):
            log.warning(f"Found class IDs outside range: {np.max(classes)}")
            classes = classes % len(CLASS_LABELS)
        
        # Apply confidence threshold
        mask = scores > self.conf_threshold
        return boxes[mask], scores[mask], classes[mask]

    def _visualizeResults(self, image, validPredictions):
        """Visualize detection results on the image."""
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.set_title(f"{self.modelName} ({self.precision}) Predictions", fontsize=16)
        # Handle different image formats
        if isinstance(image, np.ndarray):
            # If image is numpy array in RGB format
            ax.imshow(image)
        elif isinstance(image, torch.Tensor):
            # If image is a PyTorch tensor
            img_np = image.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_np)
        
        # Draw bounding boxes and labels
        for label, box, score in zip(
                validPredictions['Labels'],
                validPredictions['Boxes'],
                validPredictions['Scores']):
            
            # Extract coordinates
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            
            # Create and add rectangle
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with score
            ax.text(
                xmin-15, ymin-30,
                f'{label} ({score:.2f})',
                fontsize=12, color='white',
                bbox=dict(facecolor='green', alpha=0.5)
            )
        
        plt.axis('off')
        plt.show()
    
    def benchmark(self, loader, warmup=10, runs=100):
        """Benchmark the model on a set of images."""
        log.info(f"Benchmarking {self.modelName} TensorRT Engine with {self.precision} precision...")

        times = []
        numPredictions = []

        # Initial cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("CUDA is available. Using GPU for inference.")
        gc.collect()

        try:
            for i in range(warmup+runs):
                # Get image from dataset
                frame = loader[i % len(loader)]
                
                # Run inference
                validPredictions, inferenceTime = self.predict(frame)
                
                # Periodic cleanup during benchmark
                if i % 10 == 0 and i > 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Record metrics after warmup
                if i >= warmup:
                    times.append(inferenceTime)
                    numPredictions.append(len(validPredictions['Boxes']))
                    log.info(f"Image {i-warmup+1}/{runs}: {len(validPredictions['Boxes'])} predictions, "
                             f"inference time: {inferenceTime*1000:.4f} ms")
            
            # Ensure output directory exists
            os.makedirs('benchmarks', exist_ok=True)
            
            # Calculate statistics
            avgTime = np.mean(times)
            log.info(f"Average Inference Time: {avgTime*1000:.2f} ms over {runs} runs")
            log.info(f"Average FPS: {1/avgTime:.2f}")
            
            # Save results
            df = pd.DataFrame({
                'Image': [f"Image {i+1}" for i in range(runs)],
                'Num Predictions': numPredictions,
                'Inference Time (ms)': [t*1000 for t in times],
            })
            output_path = f"benchmarks/{self.modelName}_{self.precision}_benchmark.csv"
            df.to_csv(output_path, index=False)
            log.info(f"Benchmark results saved to {output_path}")
            
            return df
            
        except Exception as e:
            log.error(f"Error during benchmark: {e}")
            raise
            
        finally:
            # Final cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _preprocess(self, image):
        """Preprocess an image for inference."""
        # Store original dimensions
        self.orig_height, self.orig_width = image.shape[:2]
        
        # Get input shape from engine
        inp = self.inputs[0]
        shape = tuple(self.engine.get_tensor_shape(inp.name))  # e.g. (1,3,H,W)
        _, C, H, W = shape
        
        # Handle dynamic dimensions with model-specific defaults
        if H <= 0 or W <= 0:
            if self.modelName == 'rfdetr':
                H, W = 560, 560  # Default for RFDETR
            else:
                H, W = 640, 640  # Default for YOLOv11
            
            log.info(f"Using default input size for {self.modelName}: {W}x{H}")
        
        # Set shapes for dynamic input tensors
        if -1 in shape and hasattr(self.context, 'set_input_shape'):
            try:
                self.context.set_input_shape(inp.name, (1, C, H, W))
                log.info(f"Set dynamic input shape to (1, {C}, {H}, {W})")
            except Exception as e:
                log.warning(f"Could not set input shape: {e}")
        
        # Store scale factors for postprocessing
        self.scale_x = self.orig_width / W
        self.scale_y = self.orig_height / H
        
        # Resize and normalize
        resized = cv2.resize(image, (W, H)).astype(np.float32) / 255.0
        
        # HWC→CHW (height, width, channels) → (channels, height, width)
        transposed = np.transpose(resized, (2,0,1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return np.ascontiguousarray(batched)

