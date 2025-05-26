"""
TensorRT engine builder module for UAV Detection.

This module converts ONNX models to optimized TensorRT engines
for high-performance inference across different hardware platforms.
"""

import os
import sys
import logging
import argparse
import platform
import tensorrt as trt
import onnx

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")

# Constants
SUPPORTED_MODELS = ['fasterrcnn', 'yolov11', 'rfdetr']
DEFAULT_MODEL_DIR = 'models'
DEFAULT_ENGINE_DIR = 'engines'

# Define model paths mapping (similar to detector.py)
MODEL_PATHS = {
    'fasterrcnn': 'fasterrcnn.onnx',
    'yolov11': 'yolov11.onnx',
    'rfdetr': 'rfdetr.onnx',
}


class EngineBuilder:
    """
    TensorRT engine builder for converting ONNX models to optimized TensorRT engines.
    
    This class handles the complete pipeline of loading ONNX models, 
    parsing them with TensorRT, configuring optimization settings, 
    and building deployable engine files.
    """
    
    def __init__(self, verbose=False, workspace=4):
        """
        Initialize the TensorRT engine builder.
        
        Args:
            verbose (bool): Enable verbose logging for TensorRT
            workspace (int): Workspace memory size in GB
        """
        # Initialize TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE
        
        # Initialize TensorRT builder and config    
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        
        # Configure memory settings
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        # Set default optimization flags
        self.config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        self.config.set_flag(trt.BuilderFlag.FP16)
        self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        # Platform-specific optimizations
        if 'jetson' in platform.platform().lower():
            log.info("Jetson platform detected, applying Jetson-specific optimizations")
            self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            self.config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        
        self.network = None
        self.parser = None

    def build_engine(self, model_name, precision="fp16", use_int8=False):
        """
        Main pipeline for building a TensorRT engine.
        
        Args:
            model_name (str): Name of the model (without extension)
            precision (str): Inference precision ("fp16" or "fp32")
            use_int8 (bool): Whether to enable INT8 quantization
            
        Returns:
            bool: True if engine build was successful, False otherwise
        """
        self.modelName = model_name.lower()
        try:
            # Step 1: Create network
            self.create_network()
            
            # Step 2: Load ONNX model
            model = self.load_onnx(model_name)
            
            # Step 3: Parse the model
            if not self.parse_onnx(model):
                log.error("Failed to parse ONNX model")
                return False
                
            # Step 4: Set optimization profile
            self.set_optimization_profile()
            
            # Step 5: Build and save engine
            return self.create_engine(model_name, precision, use_int8)
            
        except Exception as e:
            log.error(f"Engine build process failed: {str(e)}")
            return False
    
    def create_network(self):
        """
        Initialize TensorRT network and ONNX parser.
        
        Creates a new TensorRT network with explicit batch dimension and 
        initializes an ONNX parser for the network.
        """
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
    
    def load_onnx(self, model_name):
        """
        Load and validate an ONNX model.
        
        Args:
            model_name (str): Name of the model (without extension)
            
        Returns:
            onnx.ModelProto: Loaded ONNX model
            
        Raises:
            SystemExit: If the ONNX file is not found or is invalid
        """
        # Get model path from dictionary
        model_filename = MODEL_PATHS.get(model_name.lower())
        if not model_filename:
            log.error(f"Model {model_name} not found in MODEL_PATHS dictionary")
            sys.exit(1)
            
        onnx_path = os.path.join(DEFAULT_MODEL_DIR, model_filename)
        if not os.path.exists(onnx_path):
            log.error(f"ONNX file {onnx_path} not found")
            sys.exit(1)

        log.info(f"Loading {model_name} ONNX model from {onnx_path}")
        try:
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            # Log model information
            log.info("✓ ONNX model structure is valid")
            log.info(f"✓ ONNX model topology: {len(model.graph.node)} nodes, "
                    f"{len(model.graph.input)} inputs, {len(model.graph.output)} outputs")
            
            # Log detailed input/output information
            log.info("✓ ONNX model inputs:")
            for i, input_tensor in enumerate(model.graph.input):
                shape_info = [dim.dim_value if dim.dim_value > 0 else "?" 
                            for dim in input_tensor.type.tensor_type.shape.dim]
                log.info(f"  [{i}] {input_tensor.name} - Shape: {shape_info}")
                
            log.info("✓ ONNX model outputs:")
            for i, output_tensor in enumerate(model.graph.output):
                shape_info = [dim.dim_value if dim.dim_value > 0 else "?" 
                            for dim in output_tensor.type.tensor_type.shape.dim]
                log.info(f"  [{i}] {output_tensor.name} - Shape: {shape_info}")
            
            return model
            
        except Exception as e:
            log.error(f"Failed to load ONNX model: {str(e)}")
            sys.exit(1)
    
    def parse_onnx(self, model):
        """
        Parse ONNX model with TensorRT.
        
        Args:
            model (onnx.ModelProto): ONNX model to parse
            
        Returns:
            bool: True if parsing was successful, False otherwise
        """
        log.info("Parsing ONNX model with TensorRT")
        
        if not self.parser.parse(model.SerializeToString()):
            log.error("Failed to parse ONNX model")
            for error in range(self.parser.num_errors):
                log.error(f"Parser error {error}: {self.parser.get_error(error)}")
            return False
            
        log.info("✓ Model parsed successfully")
        
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
        """
        Set optimization profile for dynamic shapes.
        
        Creates and configures an optimization profile for the network's
        input dimensions to enable efficient inference with varying batch sizes.
        """
        profile = self.builder.create_optimization_profile()
        input_tensor = self.network.get_input(0)
        input_shape = input_tensor.shape
        input_name = input_tensor.name
        
        # Handle dynamic shapes (dimensions with -1)
        has_dynamic_shape = any(dim == -1 for dim in input_shape)
        
        if has_dynamic_shape:
            log.info(f"Dynamic input shape detected: {input_shape}")
            
            # Get the actual shape, replacing -1 with concrete values
            actual_shape = []
            
            for i, dim in enumerate(input_shape):
                if dim == -1:
                    if i == 0:  # Batch dimension
                        actual_shape.append(1)  # Set min batch size to 1
                    elif i == 1:  # Channel dimension
                        actual_shape.append(3)  # RGB images have 3 channels
                    else:  # Height or width
                        # For YOLO models, set specific image size ranges
                        if self.modelName.lower() == 'yolov11':
                            actual_shape.append(640)  # Start with 640
                        else:
                            actual_shape.append(640)  # Default minimum dimension
                else:
                    actual_shape.append(dim)
            
            # Create min, optimal, max shapes
            min_shape = tuple(actual_shape)
            
            # For opt shape, increase spatial dimensions
            opt_shape = list(min_shape)
            if len(opt_shape) >= 3:  # For image inputs
                opt_shape[-2] = 1280  # Height
                opt_shape[-1] = 1280  # Width
            
            # For max shape, use the export dimensions
            max_shape = list(min_shape)
            if len(max_shape) >= 3:  # For image inputs
                max_shape[-2] = 2720  # Height
                max_shape[-1] = 2720  # Width
                
            # Convert lists to tuples
            opt_shape = tuple(opt_shape)
            max_shape = tuple(max_shape)
            
            log.info(f"Setting dynamic optimization profile for {input_name}:")
            log.info(f"  Min shape: {min_shape}")
            log.info(f"  Opt shape: {opt_shape}")
            log.info(f"  Max shape: {max_shape}")
        else:
            # Static shape - use the same for min, opt, max
            min_shape = opt_shape = max_shape = tuple(input_shape)
            log.info(f"Setting static optimization profile: {min_shape}")
        
        # Set the profile dimensions
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        self.config.add_optimization_profile(profile)
        log.info(f"Set optimization profile with shape {input_shape}")

    def create_engine(self, model_name, precision="fp16", use_int8=False):
        """
        Build and save TensorRT engine.
        
        Args:
            model_name (str): Name of the model (without extension)
            precision (str): Inference precision ("fp16" or "fp32")
            use_int8 (bool): Whether to enable INT8 quantization
            
        Returns:
            bool: True if engine was successfully built and saved
        """
        engine_path = os.path.join(DEFAULT_ENGINE_DIR, f"{model_name.lower()}_{precision}.trt")
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        
        log.info(f"Building {precision.upper()} engine")
        
        # Configure precision flags
        if precision == "fp16":
            self.config.set_flag(trt.BuilderFlag.FP16)
        else:
            # Clear FP16 flag if using FP32
            if self.config.get_flag(trt.BuilderFlag.FP16):
                self.config.clear_flag(trt.BuilderFlag.FP16)
                
        # Configure INT8 mode if requested and supported
        if use_int8:
            if self.builder.platform_has_fast_int8:
                self.config.set_flag(trt.BuilderFlag.INT8)
                log.info("Enabling INT8 precision")
            else:
                log.warning("INT8 precision requested but not supported on this platform")

        try:
            # Build the engine
            log.info("Building engine - this may take a while...")
            engine_bytes = self.builder.build_serialized_network(self.network, self.config)
            
            if engine_bytes is None:
                log.error("Engine build failed - no engine returned")
                return False
                
            # Save the engine
            with open(engine_path, "wb") as f:
                f.write(engine_bytes)
                log.info(f"✓ Engine successfully saved to {engine_path}")
                
            return True
                
        except Exception as e:
            log.error(f"Engine build failed: {str(e)}")
            return False


def main(args):
    """
    Main entry point for the TensorRT engine builder.
    
    Args:
        args: Command line arguments
    """
    builder = EngineBuilder(args.verbose, args.workspace)
    success = builder.build_engine(args.model, args.precision, args.use_int8)
    
    if success:
        log.info("Engine build completed successfully")
        sys.exit(0)
    else:
        log.error("Engine build failed")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build optimized TensorRT engines from ONNX models")
    parser.add_argument(
        "-m", "--model", 
        required=True, 
        choices=SUPPORTED_MODELS,
        help="Model to optimize"
    )
    parser.add_argument(
        "-p", "--precision", 
        default="fp16",
        choices=["fp32", "fp16"], 
        help="Inference precision"
    )
    parser.add_argument(
        "--use_int8", 
        action="store_true",
        help="Enable INT8 quantization (requires calibration data)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "-w", "--workspace", 
        type=int, 
        default=4,
        help="Workspace memory in GB (4GB is Jetson-friendly default)"
    )
    
    args = parser.parse_args()
    main(args)