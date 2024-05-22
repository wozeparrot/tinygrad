from tinygrad import Tensor
import numpy as np

if __name__ == "__main__":
  t = Tensor.randn(4096, 4096).shard(("HSA:0", "REMOTE:10.0.0.10:1234:CUDA:0"), axis=-1)
  t2 = Tensor.randn(4096, 4096).shard(("HSA:0", "REMOTE:10.0.0.10:1234:CUDA:0"), axis=None)
  t3 = t @ t2

  t3_np = t.numpy() @ t2.numpy()
  np.testing.assert_allclose(t3.numpy(), t3_np, atol=1e-3)
