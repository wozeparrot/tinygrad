import argparse, pickle, socket
from socketserver import BaseRequestHandler, TCPServer
from tinygrad import Device

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", type=int, default=1234)
  args = parser.parse_args()

  class RemoteHandler(BaseRequestHandler):
    def __init__(self, *args, **kwargs):
      self.device = Device[Device.DEFAULT]

      super().__init__(*args, **kwargs)

    def handle(self):
      print("RemoteHandler", self.request)

      programs = {}
      buffers = {}

      while True:
        cmd = self.request.recv(1)
        match cmd:
          case b"\x00": # device
            device = self.request.recv(1024).decode("utf-8")
            print(f"device {device=}")
            self.device = Device[device]
            self.request.send(b"\x00")
          case b"\x01": # synchronize
            print("synchronize")
            self.device.synchronize()
            self.request.send(b"\x00")
          case b"\x02": # allocate
            size, options = pickle.loads(self.request.recv(1024))
            print(f"allocate {size=}, {options=}")
            opaque = self.device.allocator.alloc(size, options)
            buffers[opaque.value] = bytearray(size)
            pickled = pickle.dumps(opaque)
            self.request.sendall(pickled)
          case b"\x03": # free
            opaque, options = pickle.loads(self.request.recv(1024))
            print(f"free {opaque=}, {options=}")
            self.device.allocator.free(opaque, 0, options)
            del buffers[opaque.value]
            self.request.send(b"\x00")
          case b"\x04": # copyin
            dest, nbytes = pickle.loads(self.request.recv(1024))
            print(f"copyin {dest=}, {nbytes=}")
            self.request.send(b"\x00")
            src = memoryview(buffers[dest.value])
            # recv data in chunks
            total = 0
            while total < nbytes:
              recv = self.request.recv_into(src[total:], nbytes - total)
              total += recv
            self.device.allocator.copyin(dest, src)
            self.request.send(b"\x00")
          case b"\x05": # copyout
            src, nbytes = pickle.loads(self.request.recv(1024))
            print(f"copyout {src=}, {nbytes=}")
            dest = buffers[src.value]
            self.device.allocator.copyout(memoryview(dest), src)
            self.request.sendall(dest)
          case b"\x06": # compile
            nbytes = pickle.loads(self.request.recv(1024))
            print(f"compile {nbytes=}")
            self.request.send(b"\x00")
            src_bytes = bytearray()
            while nbytes > 0:
              chunk = self.request.recv(nbytes)
              src_bytes += chunk
              nbytes -= len(chunk)
            src = src_bytes.decode("utf-8")
            compiled = self.device.compiler.compile_cached(src)
            self.request.sendall(pickle.dumps(len(compiled)))
            self.request.recv(1)
            self.request.sendall(compiled)
          case b"\x07": # load
            name, nbytes, iden = pickle.loads(self.request.recv(1024))
            print(f"load {nbytes=}")
            self.request.send(b"\x00")
            lib = bytearray()
            while nbytes > 0:
              chunk = self.request.recv(nbytes)
              lib += chunk
              nbytes -= len(chunk)
            programs[iden] = self.device.runtime(name, bytes(lib))
          case b"\x08": # run
            name, bufs, global_size, local_size, vals, wait, iden = pickle.loads(self.request.recv(4096))
            print(f"run {name=}, {global_size=}, {local_size=}, {vals=}, {wait=}, {iden=}")
            programs[iden](*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)
            self.request.send(b"\x00")
          case b"\xff": # exit
            print("exit")
            break
          case b"": break
          case _: print(f"Unknown {cmd=}")
      self.request.close()

  server = TCPServer(("0.0.0.0", args.port), RemoteHandler)
  server.allow_reuse_address = True
  server.allow_reuse_port = True
  server.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  with server: server.serve_forever()
