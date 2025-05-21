import os
import sys
import logging
import onnx
import torch
import numpy as np
from onnxsim import simplify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Diagnostic")

def diagnose_onnx_model(model_name):
    """Diagnose issues with ONNX model conversion"""
    onnx_path = os.path.join("models", f"{model_name.lower()}.onnx")
    
    if not os.path.exists(onnx_path):
        log.error(f"ONNX file {onnx_path} not found")
        return

    log.info(f"Loading {model_name} ONNX model for diagnosis")
    
    try:
        # Load the ONNX model
        model = onnx.load(onnx_path)
        
        # Basic validation
        log.info("Checking model structure...")
        onnx.checker.check_model(model)
        log.info("✓ Basic model structure is valid")
        
        # Analyze inputs and outputs
        log.info("\nModel topology:")
        print_model_info(model)
        
        # Check for problematic reshape nodes
        problematic_nodes = find_problematic_reshape_nodes(model)
        if problematic_nodes:
            log.warning(f"Found {len(problematic_nodes)} problematic reshape nodes:")
            for i, node in enumerate(problematic_nodes):
                log.warning(f"  [{i+1}] {node.name} - inputs: {node.input}")
                print_node_details(model, node)
            
            # Suggest fixes
            log.info("\nSuggested fixes:")
            log.info("1. Modify exporter.py to use fixed input dimensions instead of dynamic ones")
            log.info("2. In builder.py, set specific shapes for reshape operations")
            log.info("3. Try using the following in builder.py's create_network method:")
            print_fix_suggestion(model_name)
        else:
            log.info("✓ No problematic reshape nodes found")
            
        # Try ONNX simplification with fixed shapes
        log.info("\nAttempting ONNX simplification with fixed shapes...")
        try_simplify_with_shapes(model, onnx_path)
            
    except Exception as e:
        log.error(f"Diagnostic failed: {str(e)}")

def print_model_info(model):
    """Print model input/output information"""
    log.info("Inputs:")
    for i, input in enumerate(model.graph.input):
        shape_info = [dim.dim_value if dim.dim_value > 0 else "?" for dim in input.type.tensor_type.shape.dim]
        log.info(f"  [{i}] {input.name} - Shape: {shape_info}")
    
    log.info("Outputs:")
    for i, output in enumerate(model.graph.output):
        shape_info = [dim.dim_value if dim.dim_value > 0 else "?" for dim in output.type.tensor_type.shape.dim]
        log.info(f"  [{i}] {output.name} - Shape: {shape_info}")

def find_problematic_reshape_nodes(model):
    """Find reshape nodes with -1 dimensions that might cause problems"""
    problematic_nodes = []
    
    for node in model.graph.node:
        if node.op_type == 'Reshape':
            # Check if this reshape has a -1 dimension
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    shape_data = onnx.numpy_helper.to_array(init)
                    if -1 in shape_data:
                        problematic_nodes.append(node)
                        break
    
    return problematic_nodes

def print_node_details(model, node):
    """Print detailed information about a node"""
    # Try to find the shape tensor
    for init in model.graph.initializer:
        if init.name == node.input[1]:
            shape_data = onnx.numpy_helper.to_array(init)
            log.info(f"    Reshape dimensions: {shape_data}")
            break

def print_fix_suggestion(model_name):
    """Print code suggestion to fix the issue"""
    suggestion = """
    # In builder.py's create_network method:
    
    # Add this before parsing the ONNX model
    input_shape = (1, 3, 1530, 2720)  # Fixed batch size and image dimensions
    
    # Replace the current simplification code with:
    model_simp, check = simplify(
        model,
        input_shapes={model.graph.input[0].name: input_shape},
        skip_constant_folding=False,
        skip_shape_inference=False
    )
    """
    log.info(suggestion)

def try_simplify_with_shapes(model, onnx_path):
    """Try to simplify the model with fixed shapes"""
    try:
        # Define fixed input shapes for common image sizes
        input_name = model.graph.input[0].name
        shapes = {
            "small": (1, 3, 640, 640),
            "medium": (1, 3, 1024, 1024),
            "large": (1, 3, 1530, 2720)
        }
        
        for size_name, shape in shapes.items():
            log.info(f"Trying simplification with {size_name} input shape {shape}...")
            try:
                model_simp, check = simplify(
                    model,
                    overwrite_input_shapes={input_name: shape},
                    skip_constant_folding=False,
                    skip_shape_inference=False,
                    dynamic_input_shape=False
                )
                
                if check:
                    log.info(f"✓ Simplification successful with {size_name} shape")
                    # Save the simplified model with shape info in filename
                    simplified_path = onnx_path.replace('.onnx', f'_simplified_{size_name}.onnx')
                    onnx.save(model_simp, simplified_path)
                    log.info(f"Saved simplified model to {simplified_path}")
                else:
                    log.warning(f"× Simplification with {size_name} shape failed validation check")
            except Exception as e:
                log.warning(f"× Simplification with {size_name} shape failed: {str(e)}")
    
    except Exception as e:
        log.error(f"Simplification attempts failed: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose ONNX model issues")
    parser.add_argument("-m", "--model", required=True, 
                        choices=["fasterrcnn", "yolov11"], help="Model to diagnose")
    
    args = parser.parse_args()
    diagnose_onnx_model(args.model)

