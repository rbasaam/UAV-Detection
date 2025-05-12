import os
import sys
import logging
import argparse
import numpy as np
import tensorrt as trt
import onnx
from onnxsim import simplify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    Optimized for maximum resource usage.
    """

    def __init__(self, verbose=False, workspace=16):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in GB. Increased for maximum optimization.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        
        # Increase the workspace memory pool size for maximum performance
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, modelName):
        """
        Parse the ONNX graph—using shape inference & simplification for FasterRCNN—
        or directly for YOLOv11, and create the corresponding TensorRT network.
        :param modelName: 'fasterrcnn' or 'yolov11'
        """
        # 1) Prepare the parser & network
        self.network = self.builder.create_network(1)
        self.parser  = trt.OnnxParser(self.network, self.trt_logger)

        # 2) Locate the ONNX file
        onnx_path = os.path.realpath(os.path.join("models", f"{modelName.lower()}.onnx"))
        if not os.path.exists(onnx_path):
            log.error(f"ONNX model file {onnx_path} not found.")
            sys.exit(1)
        log.info(f"Loading ONNX model from {onnx_path}...")

        if modelName.lower() == "fasterrcnn":
            # ——————————————————————————
            # A) Load & infer shapes
            model = onnx.load(onnx_path)
            model = onnx.shape_inference.infer_shapes(model)

            # B) Simplify with an explicit input shape to remove '-1' wildcards
            input_proto = model.graph.input[0]
            input_name  = input_proto.name
            shape_hint = [1, 3, 1530, 2720]
            log.info(f"Simplifying ONNX model with shape hint {shape_hint}...")
            model_simp, check = simplify(
                model,
                overwrite_input_shapes={input_name: shape_hint},
                test_input_shapes={input_name: shape_hint}
            )
            if not check:
                log.warning("ONNX simplifier sanity check failed; proceeding anyway.")
            serialized = model_simp.SerializeToString()

            # C) Parse the simplified ONNX into TensorRT
            if not self.parser.parse(serialized):
                log.error("Failed to parse simplified FasterRCNN ONNX model.")
                for idx in range(self.parser.num_errors):
                    log.error(self.parser.get_error(idx))
                sys.exit(1)
            log.info("FasterRCNN ONNX model parsed successfully.")

        elif modelName.lower() == "yolov11":
            # ——————————————————————————
            # Direct parse, no special handling
            with open(onnx_path, "rb") as f:
                if not self.parser.parse(f.read()):
                    log.error("Failed to parse YOLOv11 ONNX model.")
                    for idx in range(self.parser.num_errors):
                        log.error(f"Error {idx}: {self.parser.get_error(idx)}")
                    sys.exit(1)
            log.info("YOLOv11 ONNX model parsed successfully.")

        else:
            log.error(f"Model '{modelName}' not supported. Only 'fasterrcnn' and 'yolov11' are allowed.")
            sys.exit(1)

        # 3) Log inputs/outputs for either branch
        inputs  = [self.network.get_input(i)  for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description:")
        for i, inp in enumerate(inputs):
            self.batch_size = inp.shape[0]
            log.info(f"  Input {i}: '{inp.name}', shape={tuple(inp.shape)}, dtype={trt.nptype(inp.dtype)}")
        for i, out in enumerate(outputs):
            log.info(f"  Output {i}: '{out.name}', shape={tuple(out.shape)}, dtype={trt.nptype(out.dtype)}")

        assert self.batch_size > 0, "Batch size must be positive after parsing."
                
                
    def create_engine(self, modelName, precision="fp16", use_int8=False):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16'.
        :param use_int8: Enable INT8 precision mode if hardware supports it.
        """

        engine_path = os.path.join("engines", f"{modelName.lower()}_{precision}.trt")
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        # Set precision flags
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.FP16)

        if use_int8:
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)

        # Build and serialize the engine
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Failed to create engine")
            sys.exit(1)

        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)


def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.model)
    builder.create_engine(args.model, args.precision, use_int8=args.use_int8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--model", help="The input ONNX model file to load", required=True, default="model.onnx")
    parser.add_argument(
        "-p", "--precision", default="fp16", choices=["fp32", "fp16"], help="The precision mode to build in, either fp32/fp16"
    )
    parser.add_argument(
        "--use_int8", action="store_true", help="Enable INT8 precision mode (only if supported by the hardware)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument(
        "-w", "--workspace", default=16, type=int, help="The max memory workspace size to allow in GB (default 16 GB for optimization)"
    )
    args = parser.parse_args()
    main(args)