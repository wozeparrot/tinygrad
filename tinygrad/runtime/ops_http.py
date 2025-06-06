import urllib, urllib.request, gzip
from tinygrad.device import Compiled, Allocator
from tinygrad.helpers import DEBUG

class HTTPDevice(Compiled):
  def __init__(self, device:str):
    url = device[len("http:"):]
    self.method, self.url = url.split("+", 1) if "+" in url else ("GET", url)
    self.gunzip = self.url.endswith(".gz")
    super().__init__(device, HTTPAllocator(self), None, None, None)

class HTTPBuffer:
  def __init__(self, device:HTTPDevice, size:int, offset=0):
    self.device, self.size, self.offset = device, size, offset
  def __repr__(self): return f"<HTTPBuffer size={self.size} offset={self.offset}>"

class HTTPAllocator(Allocator[HTTPDevice]):
  def _alloc(self, size, options):
    return HTTPBuffer(self.dev, size)
  def _free(self, opaque, options): pass

  def _copyin(self, dest:HTTPBuffer, src:memoryview):
    pass

  def _copyout(self, dest:memoryview, src:HTTPBuffer):
    headers = {
      "User-Agent": "tinygrad 0.10.3",
      "Range": f"bytes={src.offset}-{src.offset + src.size - 1}",
    }
    if DEBUG >= 2: print(f"http request: {self.dev.method} {self.dev.url} with headers {headers}")
    with urllib.request.urlopen(urllib.request.Request(self.dev.url, headers=headers, method=self.dev.method), timeout=10) as r:
      assert r.status in [200, 206], r.status
      readfile = gzip.GzipFile(fileobj=r) if self.dev.gunzip else r
      wptr = 0
      while chunk := readfile.read(16384):
        dest[wptr:wptr + len(chunk)] = chunk
        wptr += len(chunk)
