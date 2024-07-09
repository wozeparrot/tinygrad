from __future__ import annotations
import os, ctypes, contextlib, pathlib, re, fcntl, functools, mmap, struct, tempfile, hashlib, subprocess, time, array
from typing import Tuple, List, Any
from dataclasses import dataclass
from tinygrad.device import HCQCompatCompiled, HCQCompatAllocator, HCQCompatAllocRes, Compiler, CompileError, BufferOptions, hcq_profile
from tinygrad.helpers import getenv, from_mv, mv_address, init_c_struct_t, to_mv, round_up, to_char_p_p, DEBUG, prod, PROFILE
from tinygrad.renderer.cstyle import NVRenderer
from tinygrad.runtime.ops_cuda import check as cuda_check, _get_bytes, CUDACompiler, PTXCompiler, PTX
import tinygrad.runtime.autogen.nv_gpu as nv_gpu
import tinygrad.runtime.autogen.nvrtc as nvrtc
from tinygrad.renderer.assembly import PTXRenderer
import tinygrad.runtime.autogen.libc as libc
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl # noqa: F401 # pylint: disable=unused-import
if MOCKGPU:=getenv("MOCKGPU"): import extra.mockgpu.mockgpu # noqa: F401 # pylint: disable=unused-import

def nv_iowr(fd, nr, args):
  ret = fcntl.ioctl(fd, (3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

def rm_alloc(fd, clss, root, parant, params):
  made = nv_gpu.NVOS21_PARAMETERS(hRoot=root, hObjectParent=parant, hClass=clss,
                                  pAllocParms=ctypes.cast(ctypes.byref(params), ctypes.POINTER(None)) if params is not None else None) # type: ignore
  nv_iowr(fd, nv_gpu.NV_ESC_RM_ALLOC, made)
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}: {nv_gpu.nv_status_codes.get(made.status, 'Unknown error')}")
  return made

def rm_control(cmd, sttyp, fd, client, obj, **kwargs):
  made = nv_gpu.NVOS54_PARAMETERS(hClient=client, hObject=obj, cmd=cmd, paramsSize=ctypes.sizeof(params:=sttyp(**kwargs)),
                                  params=ctypes.cast(ctypes.byref(params), ctypes.POINTER(None)) if params is not None else None) # type: ignore
  nv_iowr(fd, nv_gpu.NV_ESC_RM_CONTROL, made)
  if made.status != 0: raise RuntimeError(f"rm_control returned {made.status}: {nv_gpu.nv_status_codes.get(made.status, 'Unknown error')}")
  return params

def make_rmctrl_type():
  return type("NVRMCTRL", (object,), {name[name.find("_CTRL_CMD_")+10:].lower(): functools.partial(rm_control, dt, sttyp)
    for name,dt in nv_gpu.__dict__.items() if name.find("_CTRL_CMD_")>=0 and
      (sttyp:=getattr(nv_gpu, name.replace("_CTRL_CMD_", "_CTRL_")+"_PARAMS", getattr(nv_gpu, name+"_PARAMS", None)))})
rmctrl = make_rmctrl_type()

def uvm_ioctl(cmd, sttyp, fd, **kwargs):
  ret = fcntl.ioctl(fd, cmd, made:=sttyp(**kwargs))
  if ret != 0: raise RuntimeError(f"ioctl(uvm) returned {ret}")
  if made.rmStatus != 0: raise RuntimeError(f"uvm_ioctl returned {made.rmStatus}: {nv_gpu.nv_status_codes.get(made.rmStatus, 'Unknown error')}")
  return made

def make_uvm_type():
  return type("NVUVM", (object,), {name.replace("UVM_", "").lower(): functools.partial(uvm_ioctl, dt, getattr(nv_gpu, name+"_PARAMS"))
                                   for name,dt in nv_gpu.__dict__.items() if name.startswith("UVM_") and nv_gpu.__dict__.get(name+"_PARAMS")})
uvm = make_uvm_type()

def make_qmd_struct_type():
  fields = []
  bits = [(name,dt) for name,dt in nv_gpu.__dict__.items() if name.startswith("NVC6C0_QMDV03_00") and isinstance(dt, tuple)]
  bits += [(name+f"_{i}",dt(i)) for name,dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith("NVC6C0_QMDV03_00") and callable(dt)]
  bits = sorted(bits, key=lambda x: x[1][1])
  for i,(name, data) in enumerate(bits):
    if i > 0 and (gap:=(data[1] - bits[i-1][1][0] - 1)) != 0: fields.append((f"_reserved{i}", ctypes.c_uint32, gap))
    fields.append((name.replace("NVC6C0_QMDV03_00_", "").lower(), ctypes.c_uint32, data[0]-data[1]+1))
  return init_c_struct_t(tuple(fields))
qmd_struct_t = make_qmd_struct_type()
assert ctypes.sizeof(qmd_struct_t) == 0x40 * 4

def nvmethod(subc, mthd, size, typ=2): return (typ << 28) | (size << 16) | (subc << 13) | (mthd >> 2)
def nvdata64(data): return (data >> 32, data & 0xFFFFFFFF)
def nvdata64_le(data): return (data & 0xFFFFFFFF, data >> 32)

class NVCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch, self.compile_options = arch, [f'--gpu-architecture={arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    cuda_check(nvrtc.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_nv_{self.arch}")
  def compile(self, src:str) -> bytes:
    cuda_check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog := nvrtc.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    status = nvrtc.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options]))

    if status != 0:
      raise CompileError(f"compile failed: {_get_bytes(prog, nvrtc.nvrtcGetProgramLog, nvrtc.nvrtcGetProgramLogSize, cuda_check).decode()}")
    return _get_bytes(prog, nvrtc.nvrtcGetCUBIN, nvrtc.nvrtcGetCUBINSize, cuda_check)

def jitlink_check(status):
  if status != 0: raise CompileError(f"NvJitLink Error {status}, {nvrtc.nvJitLinkResult__enumvalues.get(status, 'Unknown')}")

class NVPTXCompiler(NVCompiler):
  def compile(self, src:str) -> bytes:
    ptxsrc = src.replace("TARGET", self.arch).replace("VERSION", "7.8" if self.arch >= "sm_89" else "7.5")
    jitlink_check(nvrtc.nvJitLinkCreate(handle := nvrtc.nvJitLinkHandle(), 1, to_char_p_p([f'-arch={self.arch}'.encode()])))
    jitlink_check(nvrtc.nvJitLinkAddData(handle, nvrtc.NVJITLINK_INPUT_PTX, ptxsrc.encode(), len(ptxsrc), "<null>".encode()))
    if nvrtc.nvJitLinkComplete(handle) != 0:
      raise CompileError(f"compile failed: {_get_bytes(handle, nvrtc.nvJitLinkGetErrorLog, nvrtc.nvJitLinkGetErrorLogSize, jitlink_check).decode()}")
    return _get_bytes(handle, nvrtc.nvJitLinkGetLinkedCubin, nvrtc.nvJitLinkGetLinkedCubinSize, jitlink_check)

class HWQueue:
  def __init__(self): self.q, self.binded_device, self.cmd_offsets = [], None, [0]
  def __del__(self):
    if self.binded_device is not None:
      self.binded_device.synchronize() # Synchronize to ensure the buffer is no longer in use.
      self.binded_device._gpu_free(self.hw_page)

  def _mark_command_end(self):
    self.cmd_offsets.append(len(self.q))
    return self
  def __len__(self): return len(self.cmd_offsets) - 1

  def memory_barrier(self): return self._mark_command_end()

  def wait(self, signal, value=0):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(ctypes.addressof(from_mv(signal))), *nvdata64_le(value),
               (3 << 0) | (1 << 24)] # ACQUIRE | PAYLOAD_SIZE_64BIT
    return self._mark_command_end()

  def timestamp(self, signal): return HWQueue.signal(self, signal, timestamp=True)

  def signal(self, signal, value=0, timestamp=False):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(ctypes.addressof(from_mv(signal))), *nvdata64_le(value),
               (1 << 0) | (1 << 20) | (1 << 24) | ((1 << 25) if timestamp else 0)] # RELEASE | RELEASE_WFI | PAYLOAD_SIZE_64BIT | RELEASE_TIMESTAMP
    self.q += [nvmethod(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 1), 0x0]
    return self._mark_command_end()

  def update_signal(self, cmd_idx, signal=None, value=None): return self.update_wait(cmd_idx, signal, value) # the same offsets and commands
  def update_wait(self, cmd_idx, signal=None, value=None):
    if signal is not None: self.q[(sigoff:=self.cmd_offsets[cmd_idx]+1):sigoff+2] = array.array('I', [*nvdata64_le(mv_address(signal))])
    if value is not None: self.q[(valoff:=self.cmd_offsets[cmd_idx]+3):valoff+2] = array.array('I', [*nvdata64_le(value)])
    return self

  def bind(self, device: NVDevice):
    self.binded_device = device
    self.hw_page = device._gpu_alloc(len(self.q) * 4, map_to_cpu=True)
    hw_view = to_mv(self.hw_page.va_addr, self.hw_page.size).cast("I")
    for i, value in enumerate(self.q): hw_view[i] = value

    # From now on, the queue is on the device for faster submission.
    self.q = hw_view # type: ignore

  def _submit(self, dev, gpfifo:GPFifo):
    if len(self.q) == 0: return

    if dev == self.binded_device: cmdq_addr = self.hw_page.va_addr
    else:
      if dev.cmdq_wptr + len(self.q) * 4 > dev.cmdq_page.size:
        assert (gpfifo.ring[gpfifo.controls.GPGet] & 0xFFFFFFFFFC) >= dev.cmdq_page.va_addr + len(self.q) * 4 or \
               gpfifo.controls.GPGet == gpfifo.controls.GPPut, "cmdq overrun"
        dev.cmdq_wptr = 0

      dev.cmdq[dev.cmdq_wptr//4:dev.cmdq_wptr//4+len(self.q)] = array.array('I', self.q)
      cmdq_addr = dev.cmdq_page.va_addr+dev.cmdq_wptr
      dev.cmdq_wptr += len(self.q) * 4

    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = (cmdq_addr//4 << 2) | (len(self.q) << 42) | (1 << 41)
    gpfifo.controls.GPPut = (gpfifo.put_value + 1) % gpfifo.entries_count
    dev.gpu_mmio[0x90 // 4] = gpfifo.token
    gpfifo.put_value += 1

class HWComputeQueue(HWQueue):
  def __init__(self):
    self.cmd_idx_to_qmd, self.cmd_idx_to_global_dims, self.cmd_idx_to_local_dims = {}, {}, {}
    super().__init__()

  def copy_from_cpu(self, gpuaddr, data):
    self.q += [nvmethod(1, nv_gpu.NVC6C0_OFFSET_OUT_UPPER, 2), *nvdata64(gpuaddr)]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_LINE_LENGTH_IN, 2), len(data)*4, 0x1]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_LAUNCH_DMA, 1), 0x41]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_LOAD_INLINE_DATA, len(data), typ=6)] + list(data)
    return self._mark_command_end()

  def exec(self, prg, kernargs, global_size=(1,1,1), local_size=(1,1,1)):
    ctypes.memmove(qmd_addr:=(kernargs + round_up(prg.constbuf_0_size, 1 << 8)), ctypes.addressof(prg.qmd), 0x40 * 4)
    self.cmd_idx_to_qmd[len(self)] = qmd = qmd_struct_t.from_address(qmd_addr) # Save qmd for later update
    self.cmd_idx_to_global_dims[len(self)] = to_mv(qmd_addr + nv_gpu.NVC6C0_QMDV03_00_CTA_RASTER_WIDTH[1] // 8, 12).cast('I')
    self.cmd_idx_to_local_dims[len(self)] = to_mv(qmd_addr + nv_gpu.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0[1] // 8, 6).cast('H')

    qmd.cta_raster_width, qmd.cta_raster_height, qmd.cta_raster_depth = global_size
    qmd.cta_thread_dimension0, qmd.cta_thread_dimension1, qmd.cta_thread_dimension2 = local_size
    qmd.constant_buffer_addr_upper_0, qmd.constant_buffer_addr_lower_0 = nvdata64(kernargs)

    if (prev_qmd:=self.cmd_idx_to_qmd.get(len(self) - 1)) is None:
      self.q += [nvmethod(1, nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, 1), (1 << 12) | (1 << 4) | (1 << 0)]
      self.q += [nvmethod(1, nv_gpu.NVC6C0_SEND_PCAS_A, 0x1), qmd_addr >> 8]
      self.q += [nvmethod(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, 0x1), 9]
    else:
      prev_qmd.dependent_qmd0_pointer = qmd_addr >> 8
      prev_qmd.dependent_qmd0_action = 1
      prev_qmd.dependent_qmd0_prefetch = 1
      prev_qmd.dependent_qmd0_enable = 1
    return self._mark_command_end()

  def update_exec(self, cmd_idx, global_size, local_size):
    # Patch the exec cmd with new launch dims
    self.cmd_idx_to_global_dims[cmd_idx][:] = array.array('I', global_size)
    self.cmd_idx_to_local_dims[cmd_idx][:] = array.array('H', local_size)

  def signal(self, signal, value=0):
    if (prev_qmd:=self.cmd_idx_to_qmd.get(len(self) - 1)) is None or prev_qmd.release0_enable == 1: return super().signal(signal, value)
    prev_qmd.release0_address_upper, prev_qmd.release0_address_lower = nvdata64(ctypes.addressof(from_mv(signal)))
    prev_qmd.release0_payload_upper, prev_qmd.release0_payload_lower = nvdata64(value)
    prev_qmd.release0_enable = 1
    self.cmd_idx_to_qmd[len(self)] = prev_qmd # this command is embedded into qmd.
    return self._mark_command_end()

  def update_signal(self, cmd_idx, signal=None, value=None):
    if (qmd:=self.cmd_idx_to_qmd.get(cmd_idx)) is None: return super().update_signal(cmd_idx, signal, value)
    if signal is not None: qmd.release0_address_upper, qmd.release0_address_lower = nvdata64(ctypes.addressof(from_mv(signal)))
    if value is not None: qmd.release0_payload_upper, qmd.release0_payload_lower = nvdata64(value)
    return self

  def submit(self, dev:NVDevice): self._submit(dev, dev.compute_gpfifo)

class HWCopyQueue(HWQueue):
  def copy(self, dest, src, copy_size):
    self.q += [nvmethod(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, 4), *nvdata64(src), *nvdata64(dest)]
    self.q += [nvmethod(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, 1), copy_size]
    self.q += [nvmethod(4, nv_gpu.NVC6B5_LAUNCH_DMA, 1), 0x182] # TRANSFER_TYPE_NON_PIPELINED | DST_MEMORY_LAYOUT_PITCH | SRC_MEMORY_LAYOUT_PITCH
    return self._mark_command_end()

  def update_copy(self, cmd_idx, dest=None, src=None):
    if dest is not None: self.q[(sigoff:=self.cmd_offsets[cmd_idx]+3):sigoff+2] = array.array('I', [*nvdata64(dest)])
    if src is not None: self.q[(sigoff:=self.cmd_offsets[cmd_idx]+1):sigoff+2] = array.array('I', [*nvdata64(src)])
    return self

  def signal(self, signal, value=0):
    self.q += [nvmethod(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, 4), *nvdata64(ctypes.addressof(from_mv(signal))), value, 4]
    self.q += [nvmethod(4, nv_gpu.NVC6B5_LAUNCH_DMA, 1), 0x14]
    return self._mark_command_end()

  def update_signal(self, cmd_idx, signal=None, value=None):
    if signal is not None: self.q[(sigoff:=self.cmd_offsets[cmd_idx]+1):sigoff+2] = array.array('I', [*nvdata64(mv_address(signal))])
    if value is not None: self.q[self.cmd_offsets[cmd_idx]+3] = value
    return self

  def submit(self, dev:NVDevice): self._submit(dev, dev.dma_gpfifo)

SHT_PROGBITS, SHT_NOBITS, SHF_ALLOC, SHF_EXECINSTR = 0x1, 0x8, 0x2, 0x4
class NVProgram:
  def __init__(self, device:NVDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 6:
      try:
        fn = (pathlib.Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
        with open(fn + ".cubin", "wb") as f: f.write(lib)
        print(subprocess.check_output(["nvdisasm", fn+".cubin"]).decode('utf-8'))
      except Exception as e: print("failed to disasm cubin", str(e))

    self.rel_info, self.global_init, self.shmem_usage = None, None, 0
    constant_buffers_data = {}

    if MOCKGPU:
      self.program, self.registers_usage = memoryview(bytearray(lib) + b'\x00' * (4 - len(lib)%4)).cast("I"), 0x10
      constant_buffers_data[0] = memoryview(bytearray(0x190))
    else:
      _phoff, _shoff, _flags, _ehsize, _phentsize, _phnum, _shentsize, _shnum, _shstrndx = struct.unpack_from("<QQIHHHHHH", self.lib, 0x20)
      sections = [struct.unpack_from("<IIQQQQIIQ", self.lib, _shoff + i * _shentsize) for i in range(_shnum)]
      shstrtab = memoryview(bytearray(self.lib[sections[_shstrndx][4]:sections[_shstrndx][4]+sections[_shstrndx][5]]))
      for sh_name, sh_type, sh_flags, _, sh_offset, sh_size, _, sh_info, _ in sections:
        section_name = shstrtab[sh_name:].tobytes().split(b'\0', 1)[0].decode('utf-8')
        if sh_type == SHT_NOBITS and sh_flags & SHF_ALLOC: self.shmem_usage = sh_size
        elif sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC and sh_flags & SHF_EXECINSTR:
          self.program = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")
          self.registers_usage = sh_info >> 24
        if match := re.match(r'\.nv\.constant(\d+)', section_name):
          constant_buffers_data[int(match.group(1))] = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")
        if section_name == ".nv.global.init": self.global_init = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")
        elif section_name.startswith(".rel.text"): self.rel_info = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast('I')
        elif section_name == ".nv.info":
          section_data = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")
          for i in range(sh_size // 12):
            if section_data[i * 3 + 0] & 0xffff == 0x1204: self.device._ensure_has_local_memory(section_data[i * 3 + 2] + 0x240)

    # Registers allocation granularity per warp is 256, warp allocaiton granularity is 4. Register file size is 65536.
    self.max_threads = ((65536 // round_up(self.registers_usage * 32, 256)) // 4) * 4 * 32

    # Load program and constant buffers (if any)
    # NOTE: Ensure at least 4KB of space after the program to mitigate prefetch memory faults.
    self.lib_sz = round_up(round_up(self.program.nbytes, 128) + max(0x1000, sum([round_up(x.nbytes, 128) for i,x in constant_buffers_data.items()]) +
                           round_up(0 if self.global_init is None else self.global_init.nbytes, 128)), 0x1000)
    self.lib_gpu = self.device.allocator.alloc(self.lib_sz)

    self.constbuffer_0 = [0] * 88
    self.constbuffer_0[6:12] = [*nvdata64_le(self.device.shared_mem_window), *nvdata64_le(self.device.local_mem_window), *nvdata64_le(0xfffdc0)]

    smem_config = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1
    self.qmd = qmd_struct_t(qmd_group_id=0x3f, sm_global_caching_enable=1, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
                            invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1,
                            cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, qmd_major_version=3,
                            shared_memory_size=max(0x400, round_up(self.shmem_usage, 0x100)), min_sm_config_shared_mem_size=smem_config,
                            max_sm_config_shared_mem_size=0x1a, register_count_v=self.registers_usage, target_sm_config_shared_mem_size=smem_config,
                            barrier_count=1, shader_local_memory_high_size=self.device.slm_per_thread, program_prefetch_size=self.program.nbytes>>8,
                            program_address_lower=self.lib_gpu.va_addr&0xffffffff, program_address_upper=self.lib_gpu.va_addr>>32, sass_version=0x89,
                            program_prefetch_addr_lower_shifted=self.lib_gpu.va_addr>>8, program_prefetch_addr_upper_shifted=self.lib_gpu.va_addr>>40,
                            constant_buffer_size_shifted4_0=0x190, constant_buffer_valid_0=1, constant_buffer_invalidate_0=1)

    # NV's kernargs is constbuffer (size 0x160), then arguments to the kernel follows. Kernargs also appends QMD at the end of the kernel.
    self.constbuf_0_size = constant_buffers_data[0].nbytes if 0 in constant_buffers_data else 0
    self.kernargs_alloc_size = round_up(self.constbuf_0_size, 1 << 8) + (8 << 8)
    self.kernargs_offset = 0x160

    # constant buffer 0 is filled for each program, no need to copy it from elf (it's just zeroes)
    if 0 in constant_buffers_data: constant_buffers_data.pop(0)

    off = round_up(self.program.nbytes, 128)

    if self.rel_info is not None:
      assert self.global_init is not None
      global_init_addr = self.lib_gpu.va_addr + off
      for rel_i in range(0, len(self.rel_info), 4):
        if self.rel_info[rel_i+2] == 0x39: self.program[self.rel_info[rel_i]//4 + 1] = (global_init_addr >> 32) # R_CUDA_ABS32_HI_32
        elif self.rel_info[rel_i+2] == 0x38: self.program[self.rel_info[rel_i]//4 + 1] = (global_init_addr & 0xffffffff) # R_CUDA_ABS32_LO_32
        else: raise RuntimeError(f"unknown reloc: {self.rel_info[rel_i+2]}")

    HWComputeQueue().wait(self.device.timeline_signal, self.device.timeline_value - 1).submit(self.device)
    for st in range(0, len(self.program), 4095):
      HWComputeQueue().copy_from_cpu(self.lib_gpu.va_addr+st*4, self.program[st:st+4095]).submit(self.device)

    if self.global_init is not None:
      HWComputeQueue().copy_from_cpu(load_addr:=(self.lib_gpu.va_addr + off), self.global_init).submit(self.device)
      off += round_up(self.global_init.nbytes, 128)
      if 4 in constant_buffers_data: # >= 12.4
        # Constbuffer 4 contains a pointer to nv.global.init, load section and set up the pointer.
        assert constant_buffers_data[4].nbytes == 8
        constant_buffers_data[4][0:2] = memoryview(struct.pack('Q', load_addr)).cast('I')

    for i,data in constant_buffers_data.items():
      self.qmd.__setattr__(f'constant_buffer_addr_upper_{i}', (self.lib_gpu.va_addr + off) >> 32)
      self.qmd.__setattr__(f'constant_buffer_addr_lower_{i}', (self.lib_gpu.va_addr + off) & 0xffffffff)
      self.qmd.__setattr__(f'constant_buffer_size_shifted4_{i}', data.nbytes)
      self.qmd.__setattr__(f'constant_buffer_valid_{i}', 1)

      HWComputeQueue().copy_from_cpu(self.lib_gpu.va_addr + off, data).submit(self.device)
      off += round_up(data.nbytes, 128)

    HWComputeQueue().signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
    self.device.timeline_value += 1
    self.device.synchronize()

  def __del__(self):
    if hasattr(self, 'lib_gpu'): self.device.allocator.free(self.lib_gpu, self.lib_sz)

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if prod(local_size) > 1024 or self.max_threads < prod(local_size): raise RuntimeError("Too many resources requsted for launch")
    if any(cur > mx for cur,mx in zip(global_size, [2147483647, 65535, 65535])) or any(cur > mx for cur,mx in zip(local_size, [1024, 1024, 64])):
      raise RuntimeError("Invalid global/local dims")

    if self.device.kernargs_ptr >= (self.device.kernargs_page.va_addr + self.device.kernargs_page.size - self.kernargs_alloc_size):
      self.device.kernargs_ptr = self.device.kernargs_page.va_addr

    # HACK: Save counts of args and vars to "unused" constbuffer for later extraction in mockgpu to pass into gpuocelot.
    if MOCKGPU: self.constbuffer_0[0:2] = [len(args), len(vals)]
    kernargs = [arg_half for arg in args for arg_half in nvdata64_le(arg.va_addr)] + list(vals)

    q = HWComputeQueue().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                        .copy_from_cpu(self.device.kernargs_ptr, self.constbuffer_0 + kernargs)

    with hcq_profile(self.device, queue=q, desc=self.name, enabled=wait or PROFILE) as (sig_st, sig_en):
      q.exec(self, self.device.kernargs_ptr, global_size, local_size)

    q.signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)

    self.device.timeline_value += 1
    self.device.kernargs_ptr += self.kernargs_alloc_size

    if wait:
      self.device._wait_signal(self.device.timeline_signal, self.device.timeline_value - 1)
      if not PROFILE: self.device.signals_pool += [sig_st, sig_en]
      return (sig_en[1] - sig_st[1]) / 1e9

class NVAllocator(HCQCompatAllocator):
  def __init__(self, device:NVDevice): super().__init__(device)

  def _alloc(self, size:int, options:BufferOptions) -> HCQCompatAllocRes:
    if options.host: return self.device._gpu_host_alloc(size)
    return self.device._gpu_alloc(size, map_to_cpu=options.cpu_access, huge_page=(size > (16 << 20)))

  def _free(self, opaque, options:BufferOptions):
    self.device.synchronize()
    if options.host: self.device._gpu_host_free(opaque)
    else: self.device._gpu_free(opaque)

@dataclass
class GPFifo:
  ring: memoryview
  controls: nv_gpu.AmpereAControlGPFifo
  entries_count: int
  token: int
  put_value: int = 0

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class NVDevice(HCQCompatCompiled):
  root = None
  fd_ctl: int = -1
  fd_uvm: int = -1
  gpus_info = None
  signals_page:Any = None
  signals_pool: List[Any] = []
  uvm_vaddr: int = 0x1000000000
  host_object_enumerator: int = 0x1000
  devices: List[NVDevice] = []

  def _new_gpu_fd(self):
    fd_dev = os.open(f"/dev/nvidia{self.device_id}", os.O_RDWR | os.O_CLOEXEC)
    nv_iowr(fd_dev, nv_gpu.NV_ESC_REGISTER_FD, nv_gpu.nv_ioctl_register_fd_t(ctl_fd=self.fd_ctl))
    return fd_dev

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0, system=False):
    fd_dev = self._new_gpu_fd() if not system else os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    made = nv_gpu.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev,
      params=nv_gpu.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.device, hMemory=memory_handle, length=size, flags=flags))
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"_gpu_map_to_cpu returned {made.params.status}")
    res = libc.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), fd_dev, 0)
    os.close(fd_dev)
    return res

  def _gpu_alloc(self, size:int, contig=False, huge_page=False, va_addr=None, map_to_cpu=False, map_flags=0):
    size = round_up(size, align:=((2 << 20) if huge_page else (4 << 10)))
    alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, alignment=align, offset=0, limit=size-1, format=6, size=size,
      attr=(((nv_gpu.NVOS32_ATTR_PAGE_SIZE_HUGE << 23) if huge_page else 0) |
            ((nv_gpu.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS if contig else nv_gpu.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS) << 27)),
      attr2=((nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC << 0) | (nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_YES << 2) |
             ((nv_gpu.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB << 20) if huge_page else 0)),
      flags=(nv_gpu.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE | nv_gpu.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM | nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED |
             nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED))
    mem_handle = rm_alloc(self.fd_ctl, nv_gpu.NV1_MEMORY_USER, self.root, self.device, alloc_params).hObjectNew

    if va_addr is None: va_addr = self._alloc_gpu_vaddr(size, alignment=align)
    if map_to_cpu: va_addr = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags)
    return self._gpu_uvm_map(va_addr, size, mem_handle)

  def _gpu_system_alloc(self, size:int, va_addr=None, map_to_cpu=False, map_flags=0):
    alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, type=13,
      attr=(nv_gpu.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS << 27) | (nv_gpu.NVOS32_ATTR_LOCATION_PCI << 25),
      attr2=(nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC << 0) | (nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_NO << 2),
      flags=(nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED |
             nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED), format=6, size=size, alignment=(4<<10), offset=0, limit=size-1)
    mem_handle = rm_alloc(self.fd_ctl, nv_gpu.NV1_MEMORY_SYSTEM, self.root, self.device, alloc_params).hObjectNew

    if va_addr is None: va_addr = self._alloc_gpu_vaddr(size)
    if map_to_cpu: va_addr = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags, system=True)

    return self._gpu_uvm_map(va_addr, size, mem_handle)

  def _gpu_host_alloc(self, size):
    va_base = self._alloc_gpu_vaddr(sz:=round_up(size, 4 << 10))
    libc.mmap(va_base, sz, mmap.PROT_READ|mmap.PROT_WRITE, MAP_FIXED|mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0)
    return self._map_to_gpu(va_base, sz)

  def _gpu_free(self, mem):
    made = nv_gpu.NVOS00_PARAMETERS(hRoot=self.root, hObjectParent=self.device, hObjectOld=mem.hMemory)
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_FREE, made)
    if made.status != 0: raise RuntimeError(f"_gpu_free returned {made.status}")
    uvm.free(self.fd_uvm, base=mem.va_addr, length=mem.size)

  def _gpu_host_free(self, mem):
    uvm.free(self.fd_uvm, base=mem.va_addr, length=mem.size)
    libc.munmap(mem.va_addr, mem.size)

  def _map_to_gpu(self, va_base, size):
    NVDevice.host_object_enumerator += 1
    flags = ((nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (nv_gpu.NVOS02_FLAGS_COHERENCY_CACHED << 12) |
             (nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30))
    made = nv_gpu.nv_ioctl_nvos02_parameters_with_fd(params=nv_gpu.NVOS02_PARAMETERS(hRoot=self.root, hObjectParent=self.device, flags=flags,
      hObjectNew=NVDevice.host_object_enumerator, hClass=nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, pMemory=va_base, limit=size-1), fd=-1)
    nv_iowr(self.fd_dev, nv_gpu.NV_ESC_RM_ALLOC_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"_map_to_gpu returned {made.params.status}")
    return self._gpu_uvm_map(va_base, size, made.params.hObjectNew)

  def _gpu_uvm_map(self, va_base, size, mem_handle, create_range=True) -> nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS:
    if create_range: uvm.create_external_range(self.fd_uvm, base=va_base, length=size)
    gpu_attrs = (nv_gpu.struct_c__SA_UvmGpuMappingAttributes*256)(
      nv_gpu.struct_c__SA_UvmGpuMappingAttributes(gpuUuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), gpuMappingType = 1))

    # NOTE: va_addr is set to make rawbufs compatable with AMD.
    return uvm.map_external_allocation(self.fd_uvm, base=va_base, length=size, rmCtrlFd=self.fd_ctl, hClient=self.root, hMemory=mem_handle,
                                       gpuAttributesCount=1, perGpuAttributes=gpu_attrs, va_addr=va_base, size=size)

  def _gpu_map(self, mem):
    mem = mem._base if hasattr(mem, '_base') else mem
    if self.gpu_uuid in getattr(mem, "mapped_gpu_ids", []): return
    mem.__setattr__("mapped_gpu_ids", getattr(mem, "mapped_gpu_ids", []) + [self.gpu_uuid])
    return self._gpu_uvm_map(mem.va_addr, mem.size, mem.hMemory, create_range=False)

  def _alloc_gpu_vaddr(self, size, alignment=(4 << 10)):
    NVDevice.uvm_vaddr = (res_va:=round_up(NVDevice.uvm_vaddr, alignment)) + size
    return res_va

  def _setup_nvclasses(self):
    clsinfo = rmctrl.gpu_get_classlist_v2(self.fd_ctl, self.root, self.device)
    self.nvclasses = {clsinfo.classList[i] for i in range(clsinfo.numClasses)}
    self.compute_class = next(clss for clss in [nv_gpu.ADA_COMPUTE_A, nv_gpu.AMPERE_COMPUTE_B] if clss in self.nvclasses)

  def __init__(self, device:str=""):
    if NVDevice.root is None:
      NVDevice.fd_ctl = os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.fd_uvm = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      fd_uvm_2 = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.root = rm_alloc(self.fd_ctl, nv_gpu.NV01_ROOT_CLIENT, 0, 0, None).hObjectNew
      uvm.initialize(self.fd_uvm)
      with contextlib.suppress(RuntimeError): uvm.mm_initialize(fd_uvm_2, uvmFd=self.fd_uvm) # this error is okay, CUDA hits it too

      NVDevice.gpus_info = (nv_gpu.nv_ioctl_card_info_t*64)()
      nv_iowr(NVDevice.fd_ctl, nv_gpu.NV_ESC_CARD_INFO, NVDevice.gpus_info)

    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.fd_dev = self._new_gpu_fd()

    assert NVDevice.gpus_info[self.device_id].valid, f"No valid device found for NV:{self.device_id}. Requesting more devices than the system has?"
    gpu_info = rmctrl.gpu_get_id_info_v2(self.fd_ctl, self.root, self.root, gpuId=NVDevice.gpus_info[self.device_id].gpu_id)

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=gpu_info.deviceInstance, hClientShare=self.root,
                                                   vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES)
    self.device = rm_alloc(self.fd_ctl, nv_gpu.NV01_DEVICE_0, self.root, self.root, device_params).hObjectNew
    self.subdevice = rm_alloc(self.fd_ctl, nv_gpu.NV20_SUBDEVICE_0, self.root, self.device, None).hObjectNew
    self.usermode = rm_alloc(self.fd_ctl, nv_gpu.TURING_USERMODE_A, self.root, self.subdevice, None).hObjectNew
    self.gpu_mmio = to_mv(self._gpu_map_to_cpu(self.usermode, mmio_sz:=0x10000, flags=2), mmio_sz).cast("I")

    self._setup_nvclasses()

    rmctrl.perf_boost(self.fd_ctl, self.root, self.subdevice, duration=0xffffffff, flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) | \
      (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) | (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX << 0)))

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    vaspace = rm_alloc(self.fd_ctl, nv_gpu.FERMI_VASPACE_A, self.root, self.device, vaspace_params).hObjectNew

    raw_uuid = rmctrl.gpu_get_gid_info(self.fd_ctl, self.root, self.subdevice, flags=nv_gpu.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16)
    self.gpu_uuid = (ctypes.c_ubyte*16)(*[raw_uuid.data[i] for i in range(16)])

    uvm.register_gpu(self.fd_uvm, rmCtrlFd=-1, gpu_uuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid))
    uvm.register_gpu_vaspace(self.fd_uvm, gpuUuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), rmCtrlFd=self.fd_ctl,
                             hClient=self.root, hVaSpace=vaspace)

    for dev in self.devices:
      uvm.enable_peer_access(self.fd_uvm, gpuUuidA=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), gpuUuidB=nv_gpu.struct_nv_uuid(uuid=dev.gpu_uuid))

    if NVDevice.signals_page is None:
      NVDevice.signals_page = self._gpu_system_alloc(16 * 65536, map_to_cpu=True)
      NVDevice.signals_pool = [to_mv(self.signals_page.va_addr + off, 16).cast("Q") for off in range(0, NVDevice.signals_page.size, 16)]
    else: self._gpu_map(NVDevice.signals_page)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    channel_group = rm_alloc(self.fd_ctl, nv_gpu.KEPLER_CHANNEL_GROUP_A, self.root, self.device, channel_params).hObjectNew

    gpfifo_area = self._gpu_alloc(0x200000, contig=True, huge_page=True, map_to_cpu=True, map_flags=0x10d0000)

    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = rm_alloc(self.fd_ctl, nv_gpu.FERMI_CONTEXT_SHARE_A, self.root, channel_group, ctxshare_params).hObjectNew

    self.compute_gpfifo = self._new_gpu_fifo(gpfifo_area, ctxshare, channel_group, offset=0, entries=0x10000)
    self.dma_gpfifo = self._new_gpu_fifo(gpfifo_area, ctxshare, channel_group, offset=0x100000, entries=0x10000)

    rmctrl.gpfifo_schedule(self.fd_ctl, self.root, channel_group, bEnable=1)

    self.cmdq_page: nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS = self._gpu_alloc(0x200000, map_to_cpu=True, huge_page=True)
    self.cmdq: memoryview = to_mv(self.cmdq_page.va_addr, 0x200000).cast("I")
    self.cmdq_wptr: int = 0 # in bytes

    self.kernargs_page: nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS = self._gpu_alloc(0x4000000, map_to_cpu=True)
    self.kernargs_ptr: int = self.kernargs_page.va_addr

    sm_info = nv_gpu.NV2080_CTRL_GR_INFO(index=nv_gpu.NV2080_CTRL_GR_INFO_INDEX_SM_VERSION)
    rmctrl.gr_get_info(self.fd_ctl, self.root, self.subdevice, grInfoListSize=1, grInfoList=ctypes.addressof(sm_info))
    self.arch: str = f"sm_{(sm_info.data>>8)&0xff}{(val>>4) if (val:=sm_info.data&0xff) > 0xf else val}"

    compiler_t = (PTXCompiler if PTX else CUDACompiler) if MOCKGPU else (NVPTXCompiler if PTX else NVCompiler)
    super().__init__(device, NVAllocator(self), PTXRenderer(self.arch, device="NV") if PTX else NVRenderer(self.arch), compiler_t(self.arch),
                     functools.partial(NVProgram, self), HWComputeQueue, HWCopyQueue, timeline_signals=[self._alloc_signal(), self._alloc_signal()])

    self._cmdq_setup_compute_gpfifo()
    self._cmdq_setup_dma_gpfifo()

    NVDevice.devices.append(self)

  @classmethod
  def _read_signal(self, sig): return sig[0]

  @classmethod
  def _read_timestamp(self, sig): return sig[1]

  @classmethod
  def _set_signal(self, sig, value): sig[0] = value

  @classmethod
  def _alloc_signal(self, value=0, **kwargs) -> memoryview:
    self._set_signal(sig := self.signals_pool.pop(), value)
    return sig

  @classmethod
  def _free_signal(self, sig): self.signals_pool.append(sig)

  @classmethod
  def _wait_signal(self, signal, value=0, timeout=10000):
    start_time = time.time() * 1000
    while time.time() * 1000 - start_time < timeout:
      if signal[0] >= value: return
    raise RuntimeError(f"wait_result: {timeout} ms TIMEOUT!")

  def _gpu2cpu_time(self, gpu_time, is_copy): return self.cpu_start_time + (gpu_time - self.gpu_start_time) / 1e3

  def synchronize(self):
    NVDevice._wait_signal(self.timeline_signal, self.timeline_value - 1)
    self.cmdq_wptr = 0

    if self.timeline_value > (1 << 63): self._wrap_timeline_signal()
    if PROFILE: self._prof_process_events()

  def _new_gpu_fifo(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400) -> GPFifo:
    notifier = self._gpu_system_alloc(48 << 20)
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=notifier.hMemory, hObjectBuffer=gpfifo_area.hMemory,
      gpFifoOffset=gpfifo_area.va_addr+offset, gpFifoEntries=entries, hContextShare=ctxshare,
      hUserdMemory=(ctypes.c_uint32*8)(gpfifo_area.hMemory), userdOffset=(ctypes.c_uint64*8)(entries*8+offset))
    gpfifo = rm_alloc(self.fd_ctl, nv_gpu.AMPERE_CHANNEL_GPFIFO_A, self.root, channel_group, params).hObjectNew
    rm_alloc(self.fd_ctl, self.compute_class, self.root, gpfifo, None)
    rm_alloc(self.fd_ctl, nv_gpu.AMPERE_DMA_COPY_B, self.root, gpfifo, None)

    ws_token_params = rmctrl.gpfifo_get_work_submit_token(self.fd_ctl, self.root, gpfifo, workSubmitToken=-1)
    assert ws_token_params.workSubmitToken != -1

    channel_base = self._alloc_gpu_vaddr(0x4000000)
    uvm.register_channel(self.fd_uvm, gpuUuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), rmCtrlFd=self.fd_ctl, hClient=self.root,
                         hChannel=gpfifo, base=channel_base, length=0x4000000)

    return GPFifo(ring=to_mv(gpfifo_area.va_addr + offset, entries * 8).cast("Q"), entries_count=entries, token=ws_token_params.workSubmitToken,
                  controls=nv_gpu.AmpereAControlGPFifo.from_address(gpfifo_area.va_addr + offset + entries * 8))

  def _cmdq_setup_compute_gpfifo(self):
    # Set windows addresses to not collide with other allocated buffers.
    self.shared_mem_window, self.local_mem_window, self.slm_per_thread = 0xfe000000, 0xff000000, 0

    queue = HWComputeQueue()
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_OBJECT, 1), self.compute_class]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 2), *nvdata64(self.local_mem_window)]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2), *nvdata64(self.shared_mem_window)]
    queue.signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1

    self.synchronize()

  def _cmdq_setup_dma_gpfifo(self):
    queue = HWCopyQueue()
    queue.q += [nvmethod(4, nv_gpu.NVC6C0_SET_OBJECT, 1), nv_gpu.AMPERE_DMA_COPY_B]
    queue.signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1

    self.synchronize()

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required: return

    self.synchronize()
    if hasattr(self, 'shader_local_mem'): self._gpu_free(self.shader_local_mem) # type: ignore # pylint: disable=access-member-before-definition

    self.slm_per_thread = round_up(required, 32)
    bytes_per_warp = round_up(self.slm_per_thread * 32, 0x200)
    bytes_per_tpc = round_up(bytes_per_warp * 48 * 2, 0x8000)
    self.shader_local_mem = self._gpu_alloc(round_up(bytes_per_tpc * 64, 0x20000), huge_page=True, contig=True)

    queue = HWComputeQueue()
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, 2), *nvdata64(self.shader_local_mem.va_addr)]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 3), *nvdata64(bytes_per_tpc), 0x40]
    queue.signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1
