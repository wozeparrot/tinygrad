import torch
from typing import Dict, Callable, Optional
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, TernaryOps, Op, Interpreted
from tinygrad.helpers import getenv, dtypes, prod, DType
from tinygrad.runtime.ops_cpu import base_fxn_for_op, einsum_mulacc
from tinygrad.runtime.lib import RawBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))
type_map = {torch.float64: dtypes.float64, torch.float16: dtypes.float16, torch.float32: dtypes.float32, torch.int8: dtypes.int8, torch.int32: dtypes.int32, torch.int64: dtypes.int64, torch.uint8: dtypes.uint8, torch.bool: dtypes.bool}
inverse_type_map = {v:k for k,v in type_map.items()}

torch_fxn_for_op: Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.NOOP: lambda x: x.contiguous(), UnaryOps.SQRT: lambda x: x.sqrt(), UnaryOps.EXP2: lambda x: x.exp2(), UnaryOps.LOG2: lambda x: x.log2(), UnaryOps.SIN: torch.sin,
  UnaryOps.CAST: lambda x,y: (x.view if y[1] else x.type)(next(k for k,v in type_map.items() if v==y[0])),
  BinaryOps.MAX: torch.maximum, BinaryOps.CMPLT: lambda x,y: (x<y).type(torch.promote_types(x.dtype, y.dtype)),
  MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist]),
  TernaryOps.MULACC: einsum_mulacc(lambda s,a,b: torch.einsum(s, a.float(), b.float()).type(torch.promote_types(a.dtype, b.dtype)), lambda x: x.stride(), lambda x,s: x.expand(s)),
  TernaryOps.WHERE: lambda x, y, z: torch.where(x != 0, y, z),
  MovementOps.STRIDE: lambda x, arg: x[tuple(slice(None, None, abs(i)) for i in arg)].flip([i for i,a in enumerate(arg) if a < 0]),
  MovementOps.EXPAND: lambda x, arg: x.expand(arg), MovementOps.PERMUTE: lambda x, arg: x.permute(arg)
}}

class RawTorchBuffer(RawBuffer):
  def __init__(self, size:int, dtype:DType, buf:Optional[torch.Tensor]=None): super().__init__(size, dtype, buf if buf is not None else torch.empty([size], dtype=inverse_type_map[dtype]))
  @classmethod
  def fromCPU(cls, x):
    buf = torch.from_numpy(x).requires_grad_(False).to(device)
    return cls(prod(x.shape), type_map[buf.dtype], buf)
  def toCPU(self): return self._buf.cpu().numpy()
TorchBuffer = Interpreted(RawTorchBuffer, torch_fxn_for_op, from_underlying=lambda x: RawTorchBuffer(prod(x.shape), type_map[x.dtype], x))
