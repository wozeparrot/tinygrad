from __future__ import annotations
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
import importlib
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod, getenv, DEBUG, dtypes
from typing import List,Dict
from onnx.onnx_pb import AttributeProto, ModelProto, TensorProto
try:
  from onnx.helper import tensor_dtype_to_np_dtype
except ImportError:
  # for onnx < 1.13
  from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
  tensor_dtype_to_np_dtype = lambda x: TENSOR_TYPE_TO_NP_TYPE[x]

# global numpy cache for parameters
numpy_cache = {}
def safe_numpy(t) -> np.ndarray:
  if not isinstance(t, Tensor): return t
  global numpy_cache
  if t not in numpy_cache:
    if DEBUG >= 1:
      print("numpy cache miss", t)
    tmp = t.numpy()
    numpy_cache[t] = tmp if len(tmp.shape) else tmp.reshape(1)
  assert len(numpy_cache[t].shape) > 0
  return numpy_cache[t]

onnx_ops = importlib.import_module('extra.onnx_ops')

ONNXLIMIT = getenv("ONNXLIMIT", -1)

def get_run_onnx(onnx_model: ModelProto):
  def shape_to_tuple(s): return tuple(x.dim_value for x in s.dim)
  def buffer_parse(inp: TensorProto) -> Tensor:
    if inp.data_type in (1,10,6,7):
      # TODO: this is shared with below
      if len(inp.float_data) > 0:
        ret = Tensor(np.array(inp.float_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
      elif len(inp.int64_data) > 0:
        ret = Tensor(np.array(inp.int64_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
      elif len(inp.int32_data) > 0:
        ret = Tensor(np.array(inp.int32_data, dtype=np.int32).reshape(inp.dims), requires_grad=False)
      else:
        ret = Tensor(np.frombuffer(inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)).reshape(inp.dims).astype(np.float32).copy(), requires_grad=False)
    else:
      raise Exception(f"bad data type {inp.name} {inp.dims} {inp.data_type}")
    return ret

  def attribute_parse(a: AttributeProto) -> float | int | str | Tensor | tuple[float] | tuple[int]:
    # TODO: this is not complete, see onnx/onnx_ml_pb2.pyi for a complete list
    if a.type == AttributeProto.FLOAT: return float(a.f)
    elif a.type == AttributeProto.INT: return int(a.i)
    elif a.type == AttributeProto.STRING: return a.s.decode("utf-8")
    elif a.type == AttributeProto.TENSOR: return buffer_parse(a.t) # TENSOR
    elif a.type == AttributeProto.FLOATS: return tuple(float(x) for x in a.floats)
    elif a.type == AttributeProto.INTS: return tuple(int(x) for x in a.ints)
    else: raise Exception(f"can't parse {a.type} {a}")
  def attribute_to_dict(a: RepeatedCompositeFieldContainer[AttributeProto]): return {x.name:attribute_parse(x) for x in a}

  tensors: Dict[str, Tensor] = {}

  # get weights and biases
  for inp in onnx_model.graph.initializer:
    if len(inp.raw_data) > 0:
      tensors[inp.name] = buffer_parse(inp)
    elif len(inp.float_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.float_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
    elif len(inp.int64_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.int64_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
    else:
      print(inp.name, inp.dims, inp.data_type, len(inp.raw_data))
      print(inp)
      raise Exception("no data")
    if DEBUG >= 1:
      print("realize", inp.name)
    tensors[inp.name].realize()

  # preparse the attributes
  attribute_dict = {}
  for num,n in enumerate(onnx_model.graph.node):
    attribute_dict[num] = attribute_to_dict(n.attribute)

  onnx_model_version = onnx_model.opset_import[0].version

  def run_onnx(inputs={}, debug=False):
    if getenv("DEBUGONNX"): debug = True
    input_tensors: Dict[str,Tensor] = {}
    intermediate_tensors: Dict[str,Tensor] = {}
    output_tensor_names = [x.name for x in onnx_model.graph.output]

    # get inputs
    for inp in onnx_model.graph.input:
      if inp.name in tensors: continue
      tmp=inp.type.optional_type.elem_type.tensor_type if inp.type.HasField("optional_type") else (inp.type.sequence_type.elem_type.tensor_type if inp.type.HasField("sequence_type") else inp.type.tensor_type)
      shape = shape_to_tuple(tmp.shape)
      if len(shape) >= 1: shape = tuple([x if x != 0 else 1 for x in shape])  # replace all dynamic dims with 1 for now
      if inp.name in inputs:
        if isinstance(inputs[inp.name], Tensor):
          input_tensors[inp.name] = inputs[inp.name]
        else:
          input_tensors[inp.name] = Tensor(inputs[inp.name], requires_grad=False)
        input_shape = input_tensors[inp.name].shape
        if input_shape == (0,): raise NotImplementedError("empty tensors aren't supported in tinygrad")
        assert input_shape == shape, f"wrong shape for input {inp.name}, {input_shape} isn't {shape}"
        for _,v in input_tensors.items(): v.realize()
      else:
        raise Exception(f"no data for {inp.name} with shape {shape}")

    def fetch_tensor(x: str):
      if x in tensors: return tensors[x]
      if x in intermediate_tensors: return intermediate_tensors[x]
      if x != str(): return input_tensors[x]
      return None

    for num,n in enumerate(onnx_model.graph.node):
      inp: List[Tensor] = []
      if debug: print("inputs:")
      for x in n.input:
        t = fetch_tensor(x)
        if debug: print(f"\t{x} - {t}")
        inp.append(t)
      opt = attribute_dict[num]
      if debug: print(f"{num}: op {n.op_type} shape {[x.shape if isinstance(x, Tensor) else x for x in inp]} opt {opt}")
      # free ones
      if n.op_type == "Relu": ret = inp[0].relu()
      elif n.op_type == "Sigmoid": ret = inp[0].sigmoid()
      elif n.op_type == "Tanh": ret = inp[0].tanh()
      elif n.op_type == "MatMul": ret = inp[0].matmul(inp[1])
      # one liners
      elif n.op_type == "Elu": ret = inp[0].elu(alpha=opt.get('alpha', 1.0))
      elif n.op_type == "Concat": ret = inp[0].cat(*inp[1:], dim=opt['axis'])
      elif n.op_type == "Transpose": ret = inp[0].permute(order=opt.get('perm', list(range(len(inp[0].shape))[::-1])))
      elif n.op_type == "Squeeze": ret = inp[0].reshape([s for i,s in enumerate(inp[0].shape) if i not in opt['axes']])
      elif n.op_type == "Div":
        # in openpilot, due to SHUFFLE_PAD_OPS issues, we are spending an extra kernel
        ret = inp[0].div(inp[1])
      elif n.op_type == "Constant":
        if 'value' in opt: ret = opt['value'] # tensor
        elif 'value_float' in opt: ret = Tensor(np.array(opt['value_float'], dtype=np.float32), requires_grad=False)
        elif 'value_int' in opt: ret = Tensor(np.array(opt['value_int'], dtype=np.int64), requires_grad=False)
        elif 'value_floats' in opt: ret = Tensor(np.array(opt['value_floats'], dtype=np.float32), requires_grad=False)
        elif 'value_ints' in opt: ret = Tensor(np.array(opt['value_ints'], dtype=np.int64), requires_grad=False)
        else: raise NotImplementedError(f'Constant not implemented for {opt}')
      elif n.op_type == "Reshape": ret = inp[0].reshape([int(x) if x != 0 else inp[0].shape[i] for i,x in enumerate(safe_numpy(inp[1]))])
      elif n.op_type == "Resize":
        # TODO: this is handcoded for YOLOv8
        scales = safe_numpy(inp[2])
        assert all(int(x) == x and x >= 1 for x in scales)
        ret = inp[0].reshape([val for pair in zip(inp[0].shape, [1] * len(scales)) for val in pair])
        ret = ret.expand([val for pair in zip(inp[0].shape, [int(x) for x in scales]) for val in pair])
        ret = ret.reshape([x*y for x,y in zip(inp[0].shape, [int(x) for x in scales])])
      elif n.op_type == "Gather":
        # TODO: is this correct? seems to work for simple gather ops
        axis = opt['axis'] if 'axis' in opt else 0
        shape = list(inp[0].shape)
        indices = [shape[axis]+int(x) if x<0 else int(x) for x in safe_numpy(inp[1])]
        args = [[(0,x) if j != axis else (i,i+1) for j, x in enumerate(shape)] for i in indices]
        ret = inp[0].slice(arg=args[0]).cat(*[inp[0].slice(arg=arg) for arg in args[1:]], dim=axis)
        ret = ret.reshape([s for i,s in enumerate(shape) if i != axis]) if len(indices) == 1 else ret # squeeze if needed
      elif n.op_type in ["Add", "Sub", "Mul", "Pow"]:
        if all(isinstance(x, Tensor) for x in inp) and (len(inp[0].shape) != len(inp[1].shape)) and (prod(inp[0].shape) == prod(inp[1].shape)):
          inp[1] = inp[1].reshape(inp[0].shape)
        # TODO: is this right?
        if 'broadcast' in opt: inp[1] = inp[1].reshape([-1 if i == opt['broadcast'] else 1 for i in range(len(inp[0].shape))])
        if n.op_type == "Add": ret = inp[0] + inp[1]
        if n.op_type == "Sub": ret = inp[0] - inp[1]
        if n.op_type == "Mul": ret = inp[0] * inp[1]
        if n.op_type == "Pow": ret = (inp[0] ** inp[1]).cast(inp[0].dtype)
      elif n.op_type == "Split":
        if 'split' not in opt: opt['split'] = [int(x) for x in safe_numpy(inp[1])]  # split can be a tensor
        if 'axis' not in opt: opt['axis'] = 0
        i = 0
        arg = [(0,x) for x in inp[0].shape]
        for o,s in zip(n.output, opt['split']):
          arg[opt['axis']] = (i,i+s)
          intermediate_tensors[o] = inp[0].slice(arg=arg)
          i = i+s
        continue
      elif n.op_type == "Slice":
        assert onnx_model_version >= 10, f'only onnx version >= 10 supported for slice'
        arg = [(0,x) for x in inp[0].shape]
        starts, ends = inp[1:3]
        axes = safe_numpy(Tensor.arange(inp[0].ndim, dtype=dtypes.int32) if len(inp) <= 3 else inp[3])
        steps = safe_numpy(inp[4])[0] if len(inp) > 4 else 1
        starts, ends = safe_numpy(starts.cast(dtypes.int32)).tolist(), safe_numpy(ends.cast(dtypes.int32)).tolist() # TODO: when indexing is added use that
        for i,axis in enumerate(axes.tolist()):
          assert axis % 1 == 0
          axis = int(axis)
          arg[axis] = (starts[i] if starts[i] >= 0 else inp[0].shape[axis]+starts[i], ends[i] if ends[i] >= 0 else inp[0].shape[axis]+ends[i])
        ret = inp[0].slice(arg=arg)
      elif n.op_type == "Shrink":
        bias = opt['bias'] if 'bias' in opt else 0
        ret = (inp[0] < -opt['lambd'])*(inp[0]+bias) + (inp[0] > opt['lambd'])*(inp[0]-bias)
      elif hasattr(onnx_ops, n.op_type):
        fxn = getattr(onnx_ops, n.op_type)
        if isinstance(fxn, dict):
          for k in sorted(fxn.keys()):
            if k <= onnx_model_version:
              real_fxn = fxn[k]
        else:
          real_fxn = fxn
        ret = real_fxn(*inp, **opt)
      else:
        print("UNSUPPORTED", n.op_type, n.input, n.output)
        raise Exception(f"op_type {n.op_type} not supported")
      if not isinstance(ret, tuple): ret = (ret, )
      assert len(n.output) <= len(ret), f"expected output size must be less than {len(ret)}, it's {n.output}"
      if debug: print([x.shape if isinstance(x, Tensor) else None for x in ret])
      if debug: print("outputs:")
      for i in range(len(n.output)):
        if debug: print(f"\t{n.output[i]} - {ret[i]}")
        intermediate_tensors[n.output[i]] = ret[i]
      #print(ret[0].numpy().mean())
      if num == ONNXLIMIT:
        output_tensor_names = n.output
        break

    return {outp:intermediate_tensors[outp] for outp in output_tensor_names}
  return run_onnx
