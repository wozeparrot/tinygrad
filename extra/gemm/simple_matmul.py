import numpy as np
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
dtype = dtypes.half if getenv("HALF") else dtypes.float
N = getenv("N", 4096)
CNT = getenv("CNT", 10)
a, b = Tensor.rand(N, N, dtype=dtype).realize(), Tensor.rand(N, N, dtype=dtype).realize()
for i in range(CNT):
  c = (a @ b).realize()
comp = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
nc = c.numpy()
np.testing.assert_allclose(nc, comp, atol=1e-2, rtol=1e-2)
