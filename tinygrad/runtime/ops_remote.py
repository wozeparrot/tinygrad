from typing import Tuple, Optional
import pickle, time, socket
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.renderer.cstyle import CUDARenderer, HIPRenderer

def RemoteProgram(s: socket.socket):
  class _RemoteProgram:
    def __init__(self, name:str, lib:bytes):
      self.name, self.lib = name, lib
      s.send(b"\x07")
      s.sendall(pickle.dumps((name, len(self.lib), id(self))))
      s.recv(1)
      s.sendall(self.lib)
    def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
      st = time.perf_counter()
      s.send(b"\x08")
      s.sendall(pickle.dumps((self.name, bufs, global_size, local_size, vals, wait, id(self))))
      s.recv(1)
      return time.perf_counter() - st
  return _RemoteProgram

class RemoteCompiler(Compiler):
  def __init__(self, s: socket.socket, cachekey:Optional[str]=None):
    self.s = s
    super().__init__(cachekey)
  def compile(self, src:str) -> bytes:
    self.s.send(b"\x06")
    src_bytes = src.encode("utf-8")
    self.s.sendall(pickle.dumps(len(src_bytes)))
    self.s.recv(1)
    self.s.sendall(src_bytes)
    nbytes = pickle.loads(self.s.recv(1024))
    self.s.send(b"\x00")
    lib = bytearray()
    while nbytes > 0:
      chunk = self.s.recv(nbytes)
      lib += chunk
      nbytes -= len(chunk)
    return lib

class RemoteAllocator(Allocator):
  def __init__(self, s: socket.socket): self.s = s
  def _alloc(self, size, options):
    self.s.sendall(b"\x02")
    self.s.sendall(pickle.dumps((size, options)))
    recv = self.s.recv(4096)
    return pickle.loads(recv)
  def _free(self, opaque, options):
    try: args = pickle.dumps((opaque, options))
    except: return
    self.s.send(b"\x03")
    self.s.sendall(args)
    self.s.recv(1)
  def copyin(self, dest, src:memoryview):
    self.s.send(b"\x04")
    self.s.sendall(pickle.dumps((dest, src.nbytes)))
    self.s.recv(1)
    self.s.sendall(src.tobytes())
    self.s.recv(1)
  def copyout(self, dest:memoryview, src):
    self.s.send(b"\x05")
    self.s.sendall(pickle.dumps((src, dest.nbytes)))
    total = 0
    while total < dest.nbytes:
      recv = self.s.recv_into(dest[total:], dest.nbytes - total)
      total += recv

class RemoteDevice(Compiled):
  def __init__(self, device:str):
    device = device.lstrip("REMOTE:")
    ip = device.split(":")[0]
    port = device.split(":")[1]
    device = device[len(ip)+1+len(port)+1:]
    if device.startswith("CUDA"): renderer = CUDARenderer("sm_89")
    elif device.startswith("HSA"): renderer = HIPRenderer()
    else: raise ValueError(f"Unsupported device {device}")

    # open up socket to remote device
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    self.s.connect((ip, int(port)))
    # send device
    self.s.send(b"\x00")
    self.s.sendall(device.encode("utf-8"))
    self.s.recv(1)

    super().__init__(device, RemoteAllocator(self.s), renderer, RemoteCompiler(self.s), RemoteProgram(self.s))
  def __del__(self):
    self.s.send(b"\xff")
    self.s.close()

  def synchronize(self):
    self.s.send(b"\x01")
    self.s.recv(1)
