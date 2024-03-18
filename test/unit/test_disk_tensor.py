import pathlib, unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load, safe_save, get_state_dict, torch_load
from tinygrad.helpers import Timing, fetch, temp
from test.helpers import is_dtype_supported

def compare_weights_both(url):
  import torch
  fn = fetch(url)
  tg_weights = get_state_dict(torch_load(fn))
  torch_weights = get_state_dict(torch.load(fn, map_location=torch.device('cpu')), tensor_type=torch.Tensor)
  assert list(tg_weights.keys()) == list(torch_weights.keys())
  for k in tg_weights:
    if tg_weights[k].dtype == dtypes.bfloat16: tg_weights[k] = torch_weights[k].float() # numpy doesn't support bfloat16
    if torch_weights[k].dtype == torch.bfloat16: torch_weights[k] = torch_weights[k].float() # numpy doesn't support bfloat16
    if torch_weights[k].requires_grad: torch_weights[k] = torch_weights[k].detach()
    np.testing.assert_equal(tg_weights[k].numpy(), torch_weights[k].numpy(), err_msg=f"mismatch at {k}, {tg_weights[k].shape}")
  print(f"compared {len(tg_weights)} weights")

class TestTorchLoad(unittest.TestCase):
  # pytorch pkl format
  def test_load_enet(self): compare_weights_both("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth")
  # pytorch zip format
  def test_load_enet_alt(self): compare_weights_both("https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth")
  # pytorch zip format
  def test_load_convnext(self): compare_weights_both('https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth')

  @unittest.skipUnless(is_dtype_supported(dtypes.float16), "need float16 support")
  def test_load_llama2bfloat(self): compare_weights_both("https://huggingface.co/qazalin/bf16-lightweight/resolve/main/consolidated.00.pth?download=true")

  # pytorch tar format
  def test_load_resnet(self): compare_weights_both('https://download.pytorch.org/models/resnet50-19c8e357.pth')

test_fn = pathlib.Path(__file__).parents[2] / "weights/LLaMA/7B/consolidated.00.pth"
#test_size = test_fn.stat().st_size
test_size = 1024*1024*1024*2

# sudo su -c 'sync; echo 1 > /proc/sys/vm/drop_caches' && python3 test/unit/test_disk_tensor.py TestRawDiskBuffer.test_readinto_read_speed
@unittest.skipIf(not test_fn.exists(), "download LLaMA weights for read in speed tests")
class TestRawDiskBuffer(unittest.TestCase):
  def test_readinto_read_speed(self):
    tst = np.empty(test_size, np.uint8)
    with open(test_fn, "rb") as f:
      with Timing("copy in ", lambda et_ns: f" {test_size/et_ns:.2f} GB/s"):
        f.readinto(tst)

@unittest.skipIf(Device.DEFAULT == "WEBGPU", "webgpu doesn't support uint8 datatype")
class TestSafetensors(unittest.TestCase):
  def test_real_safetensors(self):
    import torch
    from safetensors.torch import save_file
    torch.manual_seed(1337)
    tensors = {
      "weight1": torch.randn((16, 16)),
      "weight2": torch.arange(0, 17, dtype=torch.uint8),
      "weight3": torch.arange(0, 17, dtype=torch.int32).reshape(17,1,1),
      "weight4": torch.arange(0, 2, dtype=torch.uint8),
    }
    save_file(tensors, temp("model.safetensors"))

    ret = safe_load(temp("model.safetensors"))
    for k,v in tensors.items(): np.testing.assert_array_equal(ret[k].numpy(), v.numpy())
    safe_save(ret, temp("model.safetensors_alt"))
    with open(temp("model.safetensors"), "rb") as f:
      with open(temp("model.safetensors_alt"), "rb") as g:
        assert f.read() == g.read()
    ret2 = safe_load(temp("model.safetensors_alt"))
    for k,v in tensors.items(): np.testing.assert_array_equal(ret2[k].numpy(), v.numpy())

  def test_real_safetensors_open(self):
    fn = temp("real_safe")
    state_dict = {"tmp": Tensor.rand(10,10)}
    safe_save(state_dict, fn)
    import os
    assert os.path.getsize(fn) == 8+0x40+(10*10*4)
    from safetensors import safe_open
    with safe_open(fn, framework="pt", device="cpu") as f:
      assert sorted(f.keys()) == sorted(state_dict.keys())
      for k in f.keys():
        np.testing.assert_array_equal(f.get_tensor(k).numpy(), state_dict[k].numpy())

  def test_efficientnet_safetensors(self):
    from extra.models.efficientnet import EfficientNet
    model = EfficientNet(0)
    state_dict = get_state_dict(model)
    safe_save(state_dict, temp("eff0"))
    state_dict_loaded = safe_load(temp("eff0"))
    assert sorted(state_dict_loaded.keys()) == sorted(state_dict.keys())
    for k,v in state_dict.items():
      np.testing.assert_array_equal(v.numpy(), state_dict_loaded[k].numpy())

    # load with the real safetensors
    from safetensors import safe_open
    with safe_open(temp("eff0"), framework="pt", device="cpu") as f:
      assert sorted(f.keys()) == sorted(state_dict.keys())
      for k in f.keys():
        np.testing.assert_array_equal(f.get_tensor(k).numpy(), state_dict[k].numpy())

  def test_huggingface_enet_safetensors(self):
    # test a real file
    fn = fetch("https://huggingface.co/timm/mobilenetv3_small_075.lamb_in1k/resolve/main/model.safetensors")
    state_dict = safe_load(fn)
    assert len(state_dict.keys()) == 244
    assert 'blocks.2.2.se.conv_reduce.weight' in state_dict
    assert state_dict['blocks.0.0.bn1.num_batches_tracked'].numpy() == 276570
    assert state_dict['blocks.2.0.bn2.num_batches_tracked'].numpy() == 276570

  def test_metadata(self):
    metadata = {"hello": "world"}
    safe_save({}, temp('metadata.safetensors'), metadata)
    import struct
    with open(temp('metadata.safetensors'), 'rb') as f:
      dat = f.read()
    sz = struct.unpack(">Q", dat[0:8])[0]
    import json
    assert json.loads(dat[8:8+sz])['__metadata__']['hello'] == 'world'

  def test_save_all_dtypes(self):
    for dtype in dtypes.fields().values():
      if dtype in [dtypes.bfloat16]: continue # not supported in numpy
      path = temp(f"ones.{dtype}.safetensors")
      ones = Tensor(np.random.rand(10,10), dtype=dtype)
      safe_save(get_state_dict(ones), path)
      np.testing.assert_equal(ones.numpy(), list(safe_load(path).values())[0].numpy())

  def test_load_supported_types(self):
    import torch
    from safetensors.torch import save_file
    from safetensors.numpy import save_file as np_save_file
    torch.manual_seed(1337)
    tensors = {
      "weight_F16": torch.randn((2, 2), dtype=torch.float16),
      "weight_F32": torch.randn((2, 2), dtype=torch.float32),
      "weight_U8": torch.tensor([1, 2, 3], dtype=torch.uint8),
      "weight_I8": torch.tensor([-1, 2, 3], dtype=torch.int8),
      "weight_I32": torch.tensor([-1, 2, 3], dtype=torch.int32),
      "weight_I64": torch.tensor([-1, 2, 3], dtype=torch.int64),
      "weight_F64": torch.randn((2, 2), dtype=torch.double),
      "weight_BOOL": torch.tensor([True, False], dtype=torch.bool),
      "weight_I16": torch.tensor([127, 64], dtype=torch.short),
      "weight_BF16": torch.randn((2, 2), dtype=torch.bfloat16),
    }
    save_file(tensors, temp("model.safetensors"))

    loaded = safe_load(temp("model.safetensors"))
    for k,v in loaded.items():
      if v.dtype != dtypes.bfloat16:
        assert v.numpy().dtype == tensors[k].numpy().dtype
        np.testing.assert_allclose(v.numpy(), tensors[k].numpy())

    # pytorch does not support U16, U32, and U64 dtypes.
    tensors = {
      "weight_U16": np.array([1, 2, 3], dtype=np.uint16),
      "weight_U32": np.array([1, 2, 3], dtype=np.uint32),
      "weight_U64": np.array([1, 2, 3], dtype=np.uint64),
    }
    np_save_file(tensors, temp("model.safetensors"))

    loaded = safe_load(temp("model.safetensors"))
    for k,v in loaded.items():
      assert v.numpy().dtype == tensors[k].dtype
      np.testing.assert_allclose(v.numpy(), tensors[k])

def helper_test_disk_tensor(fn, data, np_fxn, tinygrad_fxn=None):
  if tinygrad_fxn is None: tinygrad_fxn = np_fxn
  pathlib.Path(temp(fn)).unlink(missing_ok=True)
  tinygrad_tensor = Tensor(data, device="CLANG").to(f"disk:{temp(fn)}")
  numpy_arr = np.array(data)
  tinygrad_fxn(tinygrad_tensor)
  np_fxn(numpy_arr)
  np.testing.assert_allclose(tinygrad_tensor.numpy(), numpy_arr)

class TestDiskTensor(unittest.TestCase):
  def test_empty(self):
    pathlib.Path(temp("dt1")).unlink(missing_ok=True)
    Tensor.empty(100, 100, device=f"disk:{temp('dt1')}")

  def test_write_ones(self):
    pathlib.Path(temp("dt2")).unlink(missing_ok=True)

    out = Tensor.ones(10, 10, device="CLANG").contiguous()
    outdisk = out.to(f"disk:{temp('dt2')}")
    print(outdisk)
    outdisk.realize()
    del out, outdisk

    import struct
    # test file
    with open(temp("dt2"), "rb") as f:
      assert f.read() == struct.pack('<f', 1.0) * 100 == b"\x00\x00\x80\x3F" * 100

    # test load alt
    reloaded = Tensor.empty(10, 10, device=f"disk:{temp('dt2')}")
    np.testing.assert_almost_equal(reloaded.numpy(), np.ones((10, 10)))

  def test_assign_slice(self):
    def assign(x,s,y): x[s] = y
    helper_test_disk_tensor("dt3", [0,1,2,3], lambda x: assign(x, slice(0,2), [13, 12]))
    helper_test_disk_tensor("dt4", [[0,1,2,3],[4,5,6,7]], lambda x: assign(x, slice(0,1), [[13, 12, 11, 10]]))

  def test_reshape(self):
    helper_test_disk_tensor("dt5", [1,2,3,4,5], lambda x: x.reshape((1,5)))
    helper_test_disk_tensor("dt6", [1,2,3,4], lambda x: x.reshape((2,2)))

  def test_assign_to_different_dtype(self):
    # NOTE: this is similar to Y_train in fetch_cifar
    t = Tensor.empty(10, device=f'disk:{temp("dt7")}', dtype=dtypes.int64)

    for i in range(5):
      data = np.array([3, 3])
      idx = 2 * i
      t[idx:idx+2].assign(data)

    np.testing.assert_array_equal(t.numpy(), np.array([3] * 10))

  @unittest.skipIf(Device.DEFAULT == "RHIP", "no real HIP device exists in CI")
  def test_bf16_disk_write_read(self):
    t = Tensor([10000, -1, -1000, -10000, 20], dtype=dtypes.float32)
    t.to(f"disk:{temp('f32')}").realize()

    # hack to "cast" f32 -> bf16
    with open(temp('f32'), "rb") as f: dat = f.read()
    adat = b''.join([dat[i+2:i+4] for i in range(0, len(dat), 4)])
    with open(temp('bf16'), "wb") as f: f.write(adat)

    t = Tensor.empty(5, dtype=dtypes.bfloat16, device=f"disk:{temp('bf16')}").llvm_bf16_cast(dtypes.float)
    assert t.numpy().tolist() == [9984., -1, -1000, -9984, 20]

if __name__ == "__main__":
  unittest.main()
