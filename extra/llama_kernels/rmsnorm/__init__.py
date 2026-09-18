from __future__ import annotations
import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import NUM_WG, THREADS_PER_WG, alloc_like, compile_hip, dname_of

def rmsnorm_fwd(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return (x * rrms).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_fwd_fxn(x_in_p, eps, device):
  return rmsnorm_fwd(Tensor(x_in_p, device=device), eps)

def _rmsnorm_bwd(grad:UOp, call:UOp) -> tuple:
  outs = call.unbound_outputs
  x_normed = Tensor(outs[0]).float()
  do_float = Tensor(grad).float()
  d_x = Tensor(outs[1]) * (do_float - x_normed * (do_float * x_normed).mean(-1, keepdim=True))
  return (d_x.cast(call.src[1].dtype).uop,)

def rmsnorm(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_fwd_fxn(x_in.as_param(0).uop, eps, x_in.device)
  outputs = UOp.call_with_outputs((fxn[0].uop, fxn[1].uop), x_in.uop, grad_fxn=_rmsnorm_bwd)
  return Tensor(outputs[0]), Tensor(outputs[1])

@functools.cache
def _custom_gptoss_final_denom(denom:UOp, rrms:UOp, x:UOp, *, dname:str, eps:float) -> UOp:
  # custom_kernel placeholders are localized per device and may retain multiple logical row dimensions. Validate
  # that local prefix exactly while launching the HIP kernel over its flat product.
  local_x_shape, local_denom_shape, local_rrms_shape = x.shard_shape, denom.shard_shape, rrms.shard_shape
  rows, hidden = math.prod(local_x_shape[:-1]), local_x_shape[-1]
  local_abi = (rows, hidden, x.dtype, denom.dtype, rrms.dtype)
  assert local_denom_shape == local_rrms_shape == local_x_shape[:-1] and \
    local_abi == (16384, 2880, dtypes.bfloat16, dtypes.float32, dtypes.float32), \
    f"unsupported GPT-OSS final RMS local ABI {local_abi}; x={x.shape}/{x.shard_shape} axis={x.axis}, " \
    f"denom={denom.shape}/{denom.shard_shape} axis={denom.axis}, rrms={rrms.shape}/{rrms.shard_shape} axis={rrms.axis}"
  threads = 64
  zero = UOp.const(0, dtypes.int32)
  sink = UOp.sink(denom.base, rrms.base, x.base,
                  denom.index(zero).store(UOp.const(0.0, dtypes.float32)), rrms.index(zero).store(UOp.const(0.0, dtypes.float32)),
                  x.index(zero).load(),
                  UOp.special(threads, "lidx0"), UOp.special(rows // threads, "gidx0"),
                  arg=KernelInfo(f"gptoss_final_rmsnorm_denom_rrms_{rows}_{hidden}",
                                 estimates=Estimates(ops=2*rows*hidden, mem=2*rows*hidden+8*rows)))
  src = (pathlib.Path(__file__).parent/"gptoss_final_denom.cpp").read_text()
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DTHREADS={threads}", f"-DEPS_LITERAL={eps}f", "-fno-finite-math-only"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def _gptoss_final_rmsnorm_fwd(x:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor, Tensor]:
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  local_shape = x.uop.shard_shape if axis is not None else x.shape
  assert axis in (None, 0) and math.prod(local_shape[:-1]) == 16384 and local_shape[-1] == 2880, \
    f"unsupported GPT-OSS final RMS tensor ABI {x.shape}/{x.uop.shard_shape} axis={axis}"
  denom = alloc_like(x.shape[:-1], dtypes.float32, x.device, axis).clone()
  rrms = alloc_like(x.shape[:-1], dtypes.float32, x.device, axis).clone()
  denom, rrms, *_ = Tensor.custom_kernel(denom, rrms, x, fxn=functools.partial(
    _custom_gptoss_final_denom, dname=dname_of(x.device), eps=eps))
  # Preserve nn.RMSNorm's BF16 boundary before the affine multiply exactly.
  x_normed = (x.float() / denom.unsqueeze(-1)).cast(x.dtype)
  return x_normed * weight, denom, rrms

@functools.cache
def _gptoss_final_rmsnorm_fwd_fxn(x_p, weight_p, eps, device):
  return _gptoss_final_rmsnorm_fwd(Tensor(x_p, device=device), Tensor(weight_p, device=device), eps)

def _gptoss_final_rmsnorm_bwd(gradient:UOp, call:UOp, *, eps:float) -> tuple:
  x, weight = Tensor(call.src[1]), Tensor(call.src[2])
  denom = Tensor(call.unbound_outputs[1])
  rrms = Tensor(call.unbound_outputs[2])
  grad = Tensor(gradient, device=x.device)
  # Different but algebraically equivalent handwritten derivatives changed a few BF16 roundings. Rebuild the exact
  # nn.RMSNorm expression so autograd retains its original operation order, then substitute the custom producer's
  # exact saved values without changing any surrounding boundary or association.
  xf = x.float()
  rrms_ref = (xf.square().mean(-1, keepdim=True) + eps).rsqrt()
  y_ref = (xf * rrms_ref).cast(x.dtype) * weight
  d_x, d_weight = y_ref.gradient(x, weight, gradient=grad)
  denom_saved = denom.unsqueeze(-1)
  rrms_from_denom = 1.0 / denom_saved
  rrms_saved = rrms.unsqueeze(-1)
  # Autograd's SQRT derivative refers to both rsqrt and its SQRT parent. Replace both in d_x or the generic x^2
  # reduction remains live. dWeight needs only rrms, so feed it the producer's direct saved value.
  replace_dx = {rrms_ref.uop:rrms_from_denom.uop, rrms_ref.uop.src[0]:denom_saved.uop}
  replace_dw = {rrms_ref.uop:rrms_saved.uop}
  return d_x.uop.substitute(replace_dx, walk=True), d_weight.uop.substitute(replace_dw, walk=True)

def gptoss_final_rmsnorm(x:Tensor, weight:Tensor, eps:float) -> Tensor:
  """GPT-OSS final RMSNorm, with a specialized denominator pass for the training shape."""
  assert x.dtype == weight.dtype == dtypes.bfloat16 and x.shape[-1] == weight.shape[0] == 2880
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  local_shape = x.uop.shard_shape if axis is not None else x.shape
  if axis not in (None, 0) or math.prod(local_shape[:-1]) != 16384:
    # Evaluation uses 8192 local rows. Keep ordinary RMSNorm's BF16 boundary and autograd for shapes that the
    # training-only HIP producer cannot handle, rather than passing them to its strict local-ABI assertion.
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight
  fxn = _gptoss_final_rmsnorm_fwd_fxn(x.as_param(0).uop, weight.as_param(1).uop, eps, x.device)
  outputs = UOp.call_with_outputs((fxn[0].uop, fxn[1].uop, fxn[2].uop),
    x.uop, weight.uop, grad_fxn=functools.partial(_gptoss_final_rmsnorm_bwd, eps=eps))
  return Tensor(outputs[0])

# rmsnorm fused with the norm-weight multiply: y = (x*rrms)*weight, in one kernel (saves the separate `x_normed*w`
# elementwise pass). Its backward must produce BOTH d_x and d_weight (the separate mul's autograd did d_weight before).
def rmsnorm_mul_fwd(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return ((x * rrms) * weight.float()).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_mul_fwd_fxn(x_in_p, w_p, eps, device):
  return rmsnorm_mul_fwd(Tensor(x_in_p, device=device), Tensor(w_p, device=device), eps)

def _rmsnorm_mul_bwd(grad:UOp, call:UOp) -> tuple:
  if getenv("FUSED_RMSNORM_MUL_BWD", 0):
    x_u, weight_u = call.src[1:3]
    x, weight, rrms = Tensor(x_u), Tensor(weight_u), Tensor(call.unbound_outputs[1])
    g = Tensor(grad, device=x_u.device)
    assert x.dtype == weight.dtype == g.dtype == dtypes.bfloat16 and weight.shape == (x.shape[-1],)
    device, axis = x.device, (x.uop.axis if isinstance(x.device, tuple) else None)
    local_rows = math.prod(x.uop.shard_shape[:-1] if axis is not None else x.shape[:-1])
    n_partials = min(NUM_WG, local_rows)
    grad_x = alloc_like(x.shape, dtypes.bfloat16, device, axis).clone()
    partial_shape = (n_partials * len(device), x.shape[-1]) if isinstance(device, tuple) and axis is not None else (n_partials, x.shape[-1])
    grad_weight_partial = alloc_like(partial_shape, dtypes.float32, device, 0 if isinstance(device, tuple) and axis is not None else None)
    grad_weight_partial = grad_weight_partial.clone()
    residual_inputs = _gptoss_residual_inputs(x_u) if getenv("GPTOSS_RMSNORM_BWD_RESIDUAL", 0) else None
    if residual_inputs is not None:
      residual, proj_phys, bias = residual_inputs
      grad_x, grad_weight_partial, *_ = Tensor.custom_kernel(
        grad_x, grad_weight_partial, g.contiguous(), residual, proj_phys, bias, weight, rrms,
        fxn=functools.partial(_custom_rmsnorm_mul_bwd_residual, dname=dname_of(device)))
    else:
      # The GPT-OSS router emits a physical [tokens, hidden] gradient. Consume that same view;
      # forcing its logical [batch, sequence, hidden] view contiguous inserts a full activation copy.
      flat_grad = getenv("GPTOSS_RMSNORM_BWD_FLAT_GRAD", 0) and (local_rows, x.shape[-1]) == (16384, 2880)
      kernel_grad = g.reshape(-1, x.shape[-1]) if flat_grad else g.contiguous()
      grad_x, grad_weight_partial, *_ = Tensor.custom_kernel(
        grad_x, grad_weight_partial, kernel_grad, x, weight, rrms,
        fxn=functools.partial(_custom_rmsnorm_mul_bwd, dname=dname_of(device)))
    return grad_x.uop, grad_weight_partial.sum(0).cast(weight_u.dtype).uop
  x = Tensor(call.src[1]).float(); weight = Tensor(call.src[2]).float()
  rrms = Tensor(call.unbound_outputs[1])
  x_normed = x * rrms                                  # recompute unweighted normed (x is call.src[1])
  d_y = Tensor(grad).float()
  dxn = d_y * weight                                   # d/d(x_normed)
  d_x = rrms * (dxn - x_normed * (dxn * x_normed).mean(-1, keepdim=True))
  dw = d_y * x_normed
  d_weight = dw.sum(axis=tuple(range(dw.ndim - 1)))    # reduce batch/seq -> [dim] (allreduces the DP batch axis)
  return (d_x.cast(call.src[1].dtype).uop, d_weight.cast(call.src[2].dtype).uop)

def _const_tuple(u:UOp) -> tuple[int, ...]|None:
  if u.op is not Ops.STACK or any(x.op is not Ops.CONST or not isinstance(x.arg, int) for x in u.src): return None
  return tuple(x.arg for x in u.src)

def _gptoss_residual_inputs(x_u:UOp) -> tuple[Tensor, Tensor, Tensor]|None:
  # Match only bf16(residual + dense_bias(padded_projection[:,:2880], bias)). The logical projection
  # remains in x_u's autograd graph; the HIP callback receives its physical 3072-wide parent instead.
  if x_u.op is not Ops.ADD or x_u.dtype != dtypes.bfloat16 or x_u.shape[-1] != 2880 or len(x_u.src) != 2: return None
  from extra.llama_kernels.dense_bias import _dense_bias_bwd
  for i, candidate in enumerate(x_u.src):
    if candidate.op is not Ops.AFTER or candidate.src[1].op is not Ops.CALL: continue
    dense_call = candidate.src[1]
    if len(dense_call.src) != 4 or dense_call.arg.grad_fxn is not _dense_bias_bwd: continue
    if candidate is not dense_call.unbound_outputs[0]: continue
    body = dense_call.src[0]
    if body.op is not Ops.SINK or len(body.src) != 1 or body.src[0].op is not Ops.STORE: continue
    value = body.src[0].src[1]
    if value.op is not Ops.ADD or {u.arg.slot for u in value.toposort(enter_calls=False) if u.op is Ops.PARAM} != {0, 1}: continue
    residual, proj, bias = Tensor(x_u.src[1-i]), Tensor(dense_call.src[1]), Tensor(dense_call.src[2])
    if residual.shape != x_u.shape or proj.shape != x_u.shape or bias.shape != (2880,): continue
    if residual.dtype != dtypes.bfloat16 or proj.dtype != dtypes.bfloat16 or bias.dtype != dtypes.bfloat16: continue

    # matmul_mx returns reshape(shrink([rows,3072], [0:rows,0:2880])). Reject any other view instead
    # of guessing a pitch from max_numel: that was unsafe when a generic SHRINK was compacted.
    rows = math.prod(x_u.shape[:-1])
    proj_2d = proj.uop.src[0] if proj.uop.op is Ops.RESHAPE and proj.uop.src[0].shape == (rows, 2880) else proj.uop
    if proj_2d.op is not Ops.SHRINK or proj_2d.shape != (rows, 2880): continue
    if _const_tuple(proj_2d.src[1]) != (0, 0) or _const_tuple(proj_2d.src[2]) != (rows, 2880): continue
    proj_phys_u = proj_2d.src[0]
    if proj_phys_u.shape != (rows, 3072) or proj_phys_u.dtype != dtypes.bfloat16: continue
    proj_phys = Tensor(proj_phys_u)
    if proj_phys.device != residual.device or bias.device != residual.device: continue
    return residual, proj_phys, bias
  return None

@functools.cache
def _custom_rmsnorm_mul_bwd(grad_x:UOp, grad_weight_partial:UOp, grad:UOp, x:UOp, weight:UOp, rrms:UOp, *, dname:str) -> UOp:
  rows, hidden = math.prod(x.shape[:-1]), x.shape[-1]
  n_partials = grad_weight_partial.shape[0]
  # Exact GPTOSS: pair two of the 16 rows owned by each workgroup, halving barriers. Full grad-x/weight-partial bits match;
  # GPU1/7 JITBEAM3 ABBA64: 112.84/111.79us -> 98.82/97.98us (-12.43/-12.35%).
  pair_rows = getenv("GPTOSS_RMSNORM_MUL_BWD_PAIR", 0) and (rows, hidden, n_partials, THREADS_PER_WG) == (16384, 2880, 1024, 256)
  assert grad_x.shape == x.shape and grad.shape in (x.shape, (rows, hidden))
  assert grad.dtype == x.dtype == grad_x.dtype == dtypes.bfloat16
  assert weight.shape == (hidden,) and weight.dtype == dtypes.bfloat16 and rrms.shape == (*x.shape[:-1], 1)
  assert rrms.dtype == grad_weight_partial.dtype == dtypes.float32 and grad_weight_partial.shape == (n_partials, hidden)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(n_partials, "gidx0")
  zero = UOp.const(0, dtypes.int32)
  accesses = (grad_x.index(zero).store(UOp.const(0, grad_x.dtype)),
              grad_weight_partial.index(zero).store(UOp.const(0.0, dtypes.float32)),
              grad.index(zero).load(), x.index(zero).load(), weight.index(zero).load(), rrms.index(zero).load())
  sink = UOp.sink(grad_x.base, grad_weight_partial.base, grad.base, x.base, weight.base, rrms.base,
                  *accesses, threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_bwd_pair{int(pair_rows)}_{rows}_{hidden}_{n_partials}",
                                 estimates=Estimates(ops=10*rows*hidden,
                                                     mem=rows*hidden*6+rows*4+n_partials*hidden*4+hidden*2)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_bwd.cpp").read_text()
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={n_partials}", f"-DTHREADS={THREADS_PER_WG}",
             f"-DGPTOSS_PAIR_ROWS={int(pair_rows)}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_rmsnorm_mul_bwd_residual(grad_x:UOp, grad_weight_partial:UOp, grad:UOp, residual:UOp, proj_phys:UOp, bias:UOp,
                                     weight:UOp, rrms:UOp, *, dname:str) -> UOp:
  rows, hidden, pad_d = math.prod(residual.shape[:-1]), residual.shape[-1], 3072
  n_partials = grad_weight_partial.shape[0]
  assert grad.shape == residual.shape == grad_x.shape and grad.dtype == residual.dtype == grad_x.dtype == dtypes.bfloat16
  assert proj_phys.shape == (rows, pad_d) and proj_phys.dtype == dtypes.bfloat16
  assert bias.shape == weight.shape == (hidden,) and bias.dtype == weight.dtype == dtypes.bfloat16
  assert rrms.shape == (*residual.shape[:-1], 1) and rrms.dtype == grad_weight_partial.dtype == dtypes.float32
  assert grad_weight_partial.shape == (n_partials, hidden)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(n_partials, "gidx0")
  zero = UOp.const(0, dtypes.int32)
  accesses = (grad_x.index(zero).store(UOp.const(0, grad_x.dtype)),
              grad_weight_partial.index(zero).store(UOp.const(0.0, dtypes.float32)), grad.index(zero).load(),
              residual.index(zero).load(), proj_phys.index(zero).load(), bias.index(zero).load(),
              weight.index(zero).load(), rrms.index(zero).load())
  sink = UOp.sink(grad_x.base, grad_weight_partial.base, grad.base, residual.base, proj_phys.base, bias.base, weight.base, rrms.base,
                  *accesses, threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_bwd_residual_{rows}_{hidden}_{pad_d}_{n_partials}",
                                 estimates=Estimates(ops=14*rows*hidden,
                                                     mem=rows*hidden*8+rows*4+n_partials*hidden*4+hidden*4)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_bwd_residual.cpp").read_text()
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DPAD_D={pad_d}", f"-DNUM_WG={n_partials}", f"-DTHREADS={THREADS_PER_WG}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def rmsnorm_mul(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_mul_fwd_fxn(x_in.as_param(0).uop, weight.as_param(1).uop, eps, x_in.device)
  outputs = UOp.call_with_outputs((fxn[0].uop, fxn[1].uop), x_in.uop, weight.uop, grad_fxn=_rmsnorm_mul_bwd)
  return Tensor(outputs[0]), Tensor(outputs[1])
