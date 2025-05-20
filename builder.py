import os
import sys
import logging
import argparse
import platform
import numpy as np
import tensorrt as trt
import onnx
from onnxsim import simplify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineBuilder:
    """TensorRT engine builder with Jetson-specific optimizations"""
    
    def __init__(self, verbose=False, workspace=4):  # Reduced workspace for Jetson
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE
            
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        
        # Jetson-optimized memory settings
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)  # Reduced to 4GB
        )
        
        # Enable Jetson-specific optimizations
        if 'jetson' in platform.platform().lower():
            self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            self.config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            
        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, modelName):
        """Create TensorRT network with dynamic shape support"""
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        
        onnx_path = os.path.join("models", f"{modelName.lower()}.onnx")
        if not os.path.exists(onnx_path):
            log.error(f"ONNX file {onnx_path} not found")
            sys.exit(1)

        log.info(f"Loading {modelName} ONNX model")
        
        # Unified model loading with dynamic shape support
        model = onnx.load(onnx_path)
        model = onnx.shape_inference.infer_shapes(model)
        
        # Dynamic shape handling
        shape_hint = [self.batch_size, 3, 1530, 2720]  # Symbolic batch size
        model_simp, check = simplify(
            model,
            # overwrite_input_shapes={model.graph.input[0].name: shape_hint},
            # test_input_shapes={model.graph.input[0].name: shape_hint}
        )
        
        if not check:
            log.warning("ONNX simplification check failed")

        if not self.parser.parse(model_simp.SerializeToString()):
            log.error("Failed to parse ONNX model")
            for error in range(self.parser.num_errors):
                log.error(f"Parser error {error}: {self.parser.get_error(error)}")
            sys.exit(1)
            
        # Set optimization profile
        profile = self.builder.create_optimization_profile()
        input_name = self.network.get_input(0).name
        profile.set_shape(input_name, (1,3,1530,2720), (1,3,1530,2720), (1,3,1530,2720))
        self.config.add_optimization_profile(profile)

        log.info("Network parsed successfully")
        self.log_network_details()

    def log_network_details(self):
        """Log detailed network configuration"""
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        
        log.info("Network Configuration:")
        for i, inp in enumerate(inputs):
            self.batch_size = inp.shape[0]
            log.info(f"Input {i}: {inp.name} | Shape: {inp.shape} | dtype: {trt.nptype(inp.dtype)}")
        for i, out in enumerate(outputs):
            log.info(f"Output {i}: {out.name} | Shape: {out.shape} | dtype: {trt.nptype(out.dtype)}")

    def create_engine(self, modelName, precision="fp16", use_int8=False):
        """Build TensorRT engine with precision constraints"""
        engine_path = os.path.join("engines", f"{modelName.lower()}_{precision}.trt")
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        
        log.info(f"Building {precision.upper()} engine")
        
        # Precision configuration
        self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        if precision == "fp16":
            self.config.set_flag(trt.BuilderFlag.FP16)
        if use_int8 and self.builder.platform_has_fast_int8:
            self.config.set_flag(trt.BuilderFlag.INT8)
            
        # Additional performance flags
        self.config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        self.config.set_flag(trt.BuilderFlag.DIRECT_IO)

        try:
            engine_bytes = self.builder.build_serialized_network(self.network, self.config)
            if engine_bytes is None:
                raise RuntimeError("Engine build failed")
                
            with open(engine_path, "wb") as f:
                f.write(engine_bytes)
                log.info(f"Engine saved to {engine_path}")
                
        except Exception as e:
            log.error(f"Build failed: {str(e)}")
            sys.exit(1)

def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.model)
    builder.create_engine(args.model, args.precision, args.use_int8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, 
                        choices=["fasterrcnn", "yolov11"], help="Model to optimize")
    parser.add_argument("-p", "--precision", default="fp16",
                        choices=["fp32", "fp16"], help="Inference precision")
    parser.add_argument("--use_int8", action="store_true",
                        help="Enable INT8 quantization")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("-w", "--workspace", type=int, default=4,
                        help="Workspace memory in GB (Jetson-friendly default)")
    
    args = parser.parse_args()
    main(args)
