import os
import sys
import logging
import argparse
import platform
import tensorrt as trt
import onnx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineBuilder:
    """Simple TensorRT engine builder for diagnostic purposes"""
    
    def __init__(self, verbose=False, workspace=4):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE
            
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        
        # Memory settings
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        self.config.set_flag(trt.BuilderFlag.FP16)
        self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        # Optimize for Jetson if detected
        if 'jetson' in platform.platform().lower():
            self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            self.config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        
        self.network = None
        self.parser = None

    def build_engine(self, model_name, precision="fp16", use_int8=False):
        """Main pipeline for building a TensorRT engine"""
        # Step 1: Create network
        self.create_network()
        
        # Step 2: Load ONNX model (without fixing)
        model = self.load_onnx(model_name)
        
        # Step 3: Parse the model
        if not self.parse_onnx(model):
            log.error("Failed to parse ONNX model")
            sys.exit(1)
            
        # Step 4: Set optimization profile
        self.set_optimization_profile()
        
        # Step 5: Build and save engine
        self.create_engine(model_name, precision, use_int8)
    
    def create_network(self):
        """Initialize TensorRT network"""
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
    
    def load_onnx(self, model_name):
        """Load ONNX model without modifications"""
        onnx_path = os.path.join("models", f"{model_name.lower()}.onnx")
        if not os.path.exists(onnx_path):
            log.error(f"ONNX file {onnx_path} not found")
            sys.exit(1)

        log.info(f"Loading {model_name} ONNX model")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        log.info("✓ ONNX model structure is valid")
        log.info("✓ ONNX model loaded successfully")
        log.info("✓ ONNX model topology:") 
        log.info(f"Inputs: {len(model.graph.input)}")
        log.info(f"Outputs: {len(model.graph.output)}")
        log.info(f"Nodes: {len(model.graph.node)}")
        log.info("✓ ONNX model inputs and outputs:")
        for i, input in enumerate(model.graph.input):
            shape_info = [dim.dim_value if dim.dim_value > 0 else "?" for dim in input.type.tensor_type.shape.dim]
            log.info(f"  [{i}] {input.name} - Shape: {shape_info}")
        for i, output in enumerate(model.graph.output):
            shape_info = [dim.dim_value if dim.dim_value > 0 else "?" for dim in output.type.tensor_type.shape.dim]
            log.info(f"  [{i}] {output.name} - Shape: {shape_info}")
        log.info("✓ ONNX model inputs and outputs loaded successfully")

        
        return model
    
    def parse_onnx(self, model):
        """Parse ONNX model with TensorRT"""
        log.info("Parsing ONNX model with TensorRT")
        
        if not self.parser.parse(model.SerializeToString()):
            log.error("Failed to parse ONNX model")
            for error in range(self.parser.num_errors):
                log.error(f"Parser error {error}: {self.parser.get_error(error)}")
            return False
            
        log.info("Model parsed successfully")
        
        # Log network configuration
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        
        log.info("Network Configuration:")
        for i, inp in enumerate(inputs):
            log.info(f"Input {i}: {inp.name} | Shape: {inp.shape}")
        for i, out in enumerate(outputs):
            log.info(f"Output {i}: {out.name} | Shape: {out.shape}")
            
        return True
    
    def set_optimization_profile(self):
        """Set optimization profile for the network"""
        profile = self.builder.create_optimization_profile()
        input_tensor = self.network.get_input(0)
        input_shape = input_tensor.shape
        
        # Use the input shape from the network
        profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
        self.config.add_optimization_profile(profile)
        log.info(f"Set optimization profile with shape {input_shape}")

    def create_engine(self, model_name, precision="fp16", use_int8=False):
        """Build TensorRT engine"""
        engine_path = os.path.join("engines", f"{model_name.lower()}_{precision}.trt")
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        
        log.info(f"Building {precision.upper()} engine")
        
        # Precision configuration
        if precision == "fp16":
            self.config.set_flag(trt.BuilderFlag.FP16)
        if use_int8 and self.builder.platform_has_fast_int8:
            self.config.set_flag(trt.BuilderFlag.INT8)

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
    builder.build_engine(args.model, args.precision, args.use_int8)

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