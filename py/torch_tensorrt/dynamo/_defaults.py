import os
import tempfile

import torch
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype

ENABLED_PRECISIONS = {dtype.f32}
DEBUG = False
DEVICE = None
DISABLE_TF32 = False
ASSUME_DYNAMIC_SHAPE_SUPPORT = False
DLA_LOCAL_DRAM_SIZE = 1073741824
DLA_GLOBAL_DRAM_SIZE = 536870912
DLA_SRAM_SIZE = 1048576
ENGINE_CAPABILITY = EngineCapability.STANDARD
WORKSPACE_SIZE = 0
MIN_BLOCK_SIZE = 5
PASS_THROUGH_BUILD_FAILURES = False
MAX_AUX_STREAMS = None
NUM_AVG_TIMING_ITERS = 1
VERSION_COMPATIBLE = False
OPTIMIZATION_LEVEL = None
SPARSE_WEIGHTS = False
TRUNCATE_DOUBLE = False
USE_PYTHON_RUNTIME = False
USE_FAST_PARTITIONER = True
ENABLE_EXPERIMENTAL_DECOMPOSITIONS = False
MAKE_REFITABLE = False
REQUIRE_FULL_COMPILATION = False
DRYRUN = False
HARDWARE_COMPATIBLE = False
SUPPORTED_KERNEL_PRECISIONS = {dtype.f32, dtype.f16, dtype.bf16, dtype.i8, dtype.f8}
TIMING_CACHE_PATH = os.path.join(tempfile.gettempdir(), "timing_cache.bin")


def default_device() -> Device:
    return Device(gpu_id=torch.cuda.current_device())
