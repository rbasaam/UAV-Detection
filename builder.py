import os
import sys
import logging
import argparse
import platform
import numpy as np
import tensorrt as trt
import onnx
from onnxsim import simplify
import onnx.helper

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineBuilder:
    """TensorRT engine builder with robust error handling and shape fixing"""
    
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
        
        # Optimize for Jetson if detected
        if 'jetson' in platform.platform().lower():
            self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            self.config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        
        self.verbose = verbose
        self.network = None
        self.parser = None

    def build_engine(self, model_name, precision="fp16", use_int8=False):
        """Main pipeline for building a TensorRT engine"""
        # Step 1: Create network
        self.create_network()
        
        # Step 2: Load and fix ONNX model
        model = self.load_and_fix_onnx(model_name)
        
        # Step 3: Parse the fixed model
        if not self.parse_onnx(model):
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
    
    def load_and_fix_onnx(self, model_name):
        """Load, analyze, fix and optimize ONNX model"""
        onnx_path = os.path.join("models", f"{model_name.lower()}.onnx")
        if not os.path.exists(onnx_path):
            log.error(f"ONNX file {onnx_path} not found")
            sys.exit(1)

        log.info(f"Loading {model_name} ONNX model")
        
        # Load the original model
        original_model = onnx.load(onnx_path)
        
        # Apply shape inference
        try:
            model = onnx.shape_inference.infer_shapes(original_model)
            log.info("Shape inference completed successfully")
        except Exception as e:
            log.warning(f"Shape inference failed: {e}. Using original model.")
            model = original_model
        
        # Print diagnostics
        self.print_model_info(model)
        
        # Find and fix problematic reshape nodes
        fixed_model = self.fix_reshape_nodes(model)
        
        # Simplify the model
        fixed_model = self.simplify_model(fixed_model)
        
        # Save the fixed model
        fixed_path = os.path.join("models", f"{model_name.lower()}_fixed.onnx")
        onnx.save(fixed_model, fixed_path)
        log.info(f"Saved fixed model to {fixed_path}")
        
        return fixed_model
    
    def fix_reshape_nodes(self, model):
        """
        Fix problematic reshape nodes in ONNX model with unique initializer names
        """
        log.info("Fixing problematic reshape nodes...")
        fixed_model = onnx.ModelProto()
        fixed_model.CopyFrom(model)
        
        # Track all reshape nodes by name for faster lookup
        reshape_map = {}
        for i, node in enumerate(fixed_model.graph.node):
            if node.op_type == 'Reshape':
                reshape_map[node.name] = (i, node)
        
        # Fix known problematic node by name and all shape tensors [0,-1]
        modified = False
        problem_shape = np.array([0, -1], dtype=np.int64)
        fixed_shape = np.array([0, 24], dtype=np.int64)
        
        # Two-pass approach:
        # 1. First find any constant initializers with [0,-1]
        constants_to_fix = []
        for init in fixed_model.graph.initializer:
            try:
                data = onnx.numpy_helper.to_array(init)
                if data.shape == (2,) and np.array_equal(data, problem_shape):
                    constants_to_fix.append(init.name)
                    log.info(f"Found problematic shape in initializer: {init.name}")
            except:
                pass
        
        # 2. Then fix reshape nodes with unique names
        for node_name, (i, node) in reshape_map.items():
            is_roi_heads = ('roi_heads/Reshape' in node_name)
            
            # Special handling for roi_heads/Reshape nodes
            if is_roi_heads and len(node.input) > 1:
                log.info(f"Fixing problematic node: {node_name}")
                
                # Create a unique name for each tensor based on node name
                unique_suffix = node_name.replace('/', '_').replace('-', '_')
                unique_tensor_name = f"fixed_shape_{unique_suffix}"
                
                # Create the fixed shape tensor with unique name
                new_shape_tensor = onnx.helper.make_tensor(
                    name=unique_tensor_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=[2],
                    vals=fixed_shape.tolist()
                )
                
                # Add to model and update node input
                fixed_model.graph.initializer.append(new_shape_tensor)
                node.input[1] = unique_tensor_name
                modified = True
            
            # Fix any reshape node using a problematic constant
            elif len(node.input) > 1 and node.input[1] in constants_to_fix:
                log.info(f"Fixing node using problematic shape: {node_name}")
                
                # Create a unique name based on the node
                unique_suffix = node_name.replace('/', '_').replace('-', '_')
                unique_tensor_name = f"fixed_shape_{unique_suffix}"
                
                # Create a new shape tensor with fixed values
                new_tensor = onnx.helper.make_tensor(
                    name=unique_tensor_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=[2],
                    vals=fixed_shape.tolist()
                )
                
                # Add the new tensor and update the node
                fixed_model.graph.initializer.append(new_tensor)
                node.input[1] = unique_tensor_name
                modified = True
        
        if modified:
            log.info("Fixed problematic reshape nodes")
        else:
            log.warning("No problematic reshape nodes fixed - may need to verify model manually")
    
        return fixed_model
    
    def simplify_model(self, model):
        """Simplify ONNX model with fixed shapes"""
        shape_hint = (1, 3, 1530, 2720)  # Fixed input dimensions
        
        try:
            log.info(f"Simplifying model with input shape: {shape_hint}")
            model_simp, check = simplify(
                model,
                input_shapes={model.graph.input[0].name: shape_hint},
                skip_constant_folding=False,
                skip_shape_inference=False,
                dynamic_input_shape=False
            )
            
            if not check:
                log.warning("ONNX simplification check failed, using original model")
                return model
                
            log.info("Model simplified successfully")
            return model_simp
            
        except Exception as e:
            log.error(f"ONNX simplification failed: {str(e)}")
            log.info("Proceeding with the fixed model without simplification")
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
        self.log_network_details()
        return True
    
    def set_optimization_profile(self):
        """Set optimization profile for the network"""
        profile = self.builder.create_optimization_profile()
        input_name = self.network.get_input(0).name
        input_shape = (1, 3, 1530, 2720)  # Fixed input shape
        
        profile.set_shape(input_name, input_shape, input_shape, input_shape)
        self.config.add_optimization_profile(profile)
        log.info(f"Set optimization profile with shape {input_shape}")

    def create_engine(self, model_name, precision="fp16", use_int8=False):
        """Build TensorRT engine with precision constraints"""
        engine_path = os.path.join("engines", f"{model_name.lower()}_{precision}.trt")
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
    
    def print_model_info(self, model):
        """Print comprehensive model information for diagnostics"""
        log.info("Model Information:")
        
        # Basic model info
        log.info("Inputs:")
        for i, input in enumerate(model.graph.input):
            shape_info = [dim.dim_value if dim.dim_value > 0 else "?" for dim in input.type.tensor_type.shape.dim]
            log.info(f"\t[{i}] {input.name}\tShape: {shape_info}")
        
        log.info("Outputs:")
        for i, output in enumerate(model.graph.output):
            shape_info = [dim.dim_value if dim.dim_value > 0 else "?" for dim in output.type.tensor_type.shape.dim]
            log.info(f"\t[{i}] {output.name}\tShape: {shape_info}")

        log.info(f"Total nodes: {len(model.graph.node)}")
        log.info(f"Total initializers: {len(model.graph.initializer)}")
        
        # Count different op types
        op_types = {}
        for node in model.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
        
        log.info(f"Operation types: {op_types}")
        
        # Detailed reshape node analysis
        reshape_nodes = []
        problematic_nodes = []
        
        log.info("Analyzing Reshape nodes...")
        for i, node in enumerate(model.graph.node):
            if node.op_type == 'Reshape':
                reshape_nodes.append(node)
                
                if 'roi_heads/Reshape' in node.name:
                    log.info(f"\t[{i}] ⚠️ Critical Reshape Node: {node.name}")
                else:
                    log.info(f"\t[{i}] Reshape Node: {node.name}")
                
                log.info(f"\t    Inputs: {[inp for inp in node.input]}")
                
                # Look for shape tensor - first in initializers, then computed
                shape_found = False
                computed_shape = False
                
                if len(node.input) > 1:
                    shape_tensor_name = node.input[1]
                    
                    # Check initializers
                    for init in model.graph.initializer:
                        if init.name == shape_tensor_name:
                            shape_found = True
                            shape_data = onnx.numpy_helper.to_array(init)
                            log.info(f"\t    Reshape Dimensions (from initializer): {shape_data}")
                            
                            # Check for problematic dimensions
                            if -1 in shape_data:
                                problematic_nodes.append(node)
                                log.warning(f"\t    ⚠️ Dynamic dimension (-1) found")
                            if 0 in shape_data and -1 in shape_data:
                                log.warning(f"\t    ⚠️ Potential TensorRT issue: [0,-1] combination")
                            
                            break
                    
                    # If not in initializers, check if it's computed by another op
                    if not shape_found:
                        for producer_node in model.graph.node:
                            if shape_tensor_name in [output for output in producer_node.output]:
                                shape_found = True
                                computed_shape = True
                                log.info(f"\t    Shape is computed by: {producer_node.op_type} node {producer_node.name}")
                                break
                
                if not shape_found:
                    log.warning(f"\t    ⚠️ Could not find shape tensor for this node")
                elif computed_shape:
                    log.warning(f"\t    ⚠️ Shape tensor is dynamically computed (potential issue for TensorRT)")
        
        log.info(f"Found {len(reshape_nodes)} Reshape nodes total")
        
        if problematic_nodes:
            log.warning(f"{len(problematic_nodes)} Known problematic nodes detected")
            for i, node in enumerate(problematic_nodes):
                log.warning(f"\t[{i+1}] {node.name}\tinputs: {node.input}")
                for init in model.graph.initializer:
                    if init.name == node.input[1]:
                        shape_data = onnx.numpy_helper.to_array(init)
                        log.info(f"\t    Reshape Dimensions: {shape_data}")
                        break

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