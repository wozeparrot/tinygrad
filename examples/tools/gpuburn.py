import time
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.helpers import GlobalCounters, getenv

GPUS = getenv("GPUS", 4) # TODO: expose a way in tinygrad to access this
N = getenv("N", 6144)
ROUNDS = getenv("ROUNDS", 8)

@TinyJit
def many_matmul(A, B):
  out = A
  for _ in range(ROUNDS): out = out@B
  return out

if __name__ == "__main__":
  A = Tensor.ones(GPUS, N, N, dtype=dtypes.half).shard(devices=tuple([f"{Device.DEFAULT}:{i}" for i in range(GPUS)]), axis=0).contiguous()
  B = Tensor.ones(GPUS, N, N, dtype=dtypes.half).shard(devices=tuple([f"{Device.DEFAULT}:{i}" for i in range(GPUS)]), axis=0).contiguous()
  while 1:
    for _ in range(10):
      GlobalCounters.reset()
      st = time.perf_counter()
      many_matmul(A, B)
      et = time.perf_counter()
    gflops = GlobalCounters.global_ops / (et - st) / 1e9
    print(f"{gflops:.2f} GFLOPS")
