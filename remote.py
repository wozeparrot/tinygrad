import argparse, pickle, socket
from socketserver import BaseRequestHandler, TCPServer
from tinygrad import Device
from tinygrad.device import CompileError

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
      opaques = {}

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
            opaques[ptr := id(opaque)] = opaque
            buffers[ptr] = bytearray(size)
            pickled = pickle.dumps(ptr)
            self.request.sendall(pickled)
          case b"\x03": # free
            ptr, options = pickle.loads(self.request.recv(1024))
            print(f"free {ptr=}, {options=}")
            del buffers[ptr]
            opaque = opaques[ptr]
            self.device.allocator.free(opaque, 0, options)
            del opaques[ptr]
            self.request.send(b"\x00")
          case b"\x04": # copyin
            ptr = int.from_bytes(self.request.recv(8), "little")
            print(f"copyin {ptr=}")
            src = memoryview(buffers[ptr])
            total = 0
            while total < src.nbytes:
              recv = self.request.recv_into(src[total:], src.nbytes - total)
              total += recv
            self.device.allocator.copyin(opaques[ptr], src)
          case b"\x05": # copyout
            ptr = int.from_bytes(self.request.recv(8), "little")
            print(f"copyout {ptr=}")
            dest = buffers[ptr]
            self.device.allocator.copyout(memoryview(dest), opaques[ptr])
            self.request.sendall(dest)
          case b"\x06": # compile
            nbytes = int.from_bytes(self.request.recv(4), "little")
            print(f"compile {nbytes=}")
            src_bytes = bytearray(nbytes)
            src_view = memoryview(src_bytes)
            total = 0
            while total < nbytes:
              recv = self.request.recv_into(src_view[total:], nbytes - total)
              total += recv
            src = src_bytes.decode("utf-8")
            try: compiled = self.device.compiler.compile(src)
            except CompileError: compiled = b""
            self.request.sendall(len(compiled).to_bytes(4, "little") + compiled)
          case b"\x07": # load
            name, nbytes, iden = pickle.loads(self.request.recv(1024))
            print(f"load {name=}, {nbytes=}, {iden=}")
            self.request.send(b"\x00")
            lib = bytearray(nbytes)
            lib_view = memoryview(lib)
            total = 0
            while total < nbytes:
              recv = self.request.recv_into(lib_view[total:], nbytes - total)
              total += recv
            programs[iden] = self.device.runtime(name, bytes(lib))
            self.request.send(b"\x00")
          case b"\x08": # run
            name, buf_ptrs, global_size, local_size, vals, wait, iden = pickle.loads(self.request.recv(4096))
            print(f"run {name=}, {global_size=}, {local_size=}, {vals=}, {wait=}, {iden=}")
            bufs = [opaques[ptr] for ptr in buf_ptrs]
            try: programs[iden](*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)
            except: failed = 1
            else: failed = 0
            self.request.send(bytes([failed]))
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
  print(f"Listening on {server.server_address}")
  with server: server.serve_forever()
