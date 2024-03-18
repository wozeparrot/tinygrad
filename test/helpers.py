import sys
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import JITRunner
from tinygrad.dtype import DType
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import Context, CI, OSX

def derandomize_model(model):
  with Context(GRAPH=0):
    for p in get_parameters(model):
      p.lazydata = Tensor.empty(p.shape, device=p.device, dtype=p.dtype).lazydata
      p.realize()

def assert_jit_cache_len(fxn, expected_len):
  assert len(fxn.jit_cache) > 0
  # until we have a better way of typing the prg in JitItem
  if issubclass(type(fxn.jit_cache[0].prg), JITRunner) and not type(fxn.jit_cache[0].prg).__name__.endswith('Graph'):
    assert len(fxn.jit_cache) == expected_len
  else:
    assert len(fxn.jit_cache) == 1
    # until we have a better way of typing the prg in JitItem
    assert type(fxn.jit_cache[0].prg).__name__.endswith('Graph')
    assert len(fxn.jit_cache[0].prg.jit_cache) == expected_len

def is_dtype_supported(dtype: DType, device: str = Device.DEFAULT):
  if dtype == dtypes.bfloat16:
    # NOTE: this requires bf16 buffer support
    return device in {"RHIP", "HSA"}
  if device in ["WEBGPU", "WEBGL"]: return dtype in [dtypes.float, dtypes.int32, dtypes.uint32]
  # for CI GPU, cl_khr_fp16 isn't supported
  # for CI LLVM, it segfaults because it can't link to the casting function
  # CUDACPU architecture is sm_35 but we need at least sm_70 to run fp16 ALUs
  # PYTHON supports half memoryview in 3.12+ https://github.com/python/cpython/issues/90751
  if dtype == dtypes.half:
    if device in ["GPU", "LLVM", "CUDA"]: return not CI
    if device == "PYTHON": return sys.version_info >= (3, 12)
  if dtype == dtypes.float64: return device != "METAL" and not (OSX and device == "GPU")
  return True
