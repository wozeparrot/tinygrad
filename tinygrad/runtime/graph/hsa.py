import ctypes, collections, time, itertools
from typing import List, Any, Dict, cast, Optional, Union, Tuple
from tinygrad.helpers import GraphException, init_c_var, round_up
from tinygrad.device import Compiled, Buffer, CompiledASTRunner, BufferXfer, MultiDeviceJITGraph, update_stats
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_hsa import HSADevice, PROFILE, Profiler
from tinygrad.features.jit import JitItem, get_input_replace, get_jit_stats, \
                                  get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.runtime.driver.hsa import check, AQLQueue, AQL_PACKET_SIZE, EMPTY_SIGNAL

def dedup_signals(signals): return [hsa.hsa_signal_t(hndl) for hndl in set([x.handle for x in signals if isinstance(x, hsa.hsa_signal_t)])]

class VirtAQLQueue(AQLQueue):
  def __init__(self, device, sz):
    self.device = device
    self.virt_queue = (hsa.hsa_kernel_dispatch_packet_t * sz)()
    self.queue_base = self.write_addr = ctypes.addressof(self.virt_queue)
    self.packets_count = 0
    self.available_packet_slots = sz
  def _wait_queue(self, need_packets=1): assert False, f"VirtQueue is too small to handle {self.packets_count+need_packets} packets!"
  def _submit_packet(self):
    self.write_addr += AQL_PACKET_SIZE
    self.packets_count += 1
    self.available_packet_slots -= 1
  def _alloc_signal(self, reusable=False): return init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(x))))

class HSAGraph(MultiDeviceJITGraph):
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache) #type:ignore
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)

    # Check all jit items are compatible.
    compiled_devices = set()
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledASTRunner): compiled_devices.add(ji.prg.device)
      elif isinstance(ji.prg, BufferXfer):
        for x in ji.rawbufs[0:2]: compiled_devices.add(cast(Buffer, x).d)
      else: raise GraphException
    if any(not isinstance(d, HSADevice) for d in compiled_devices): raise GraphException

    self.devices: List[HSADevice] = list(compiled_devices) #type:ignore

    # Allocate kernel args.
    kernargs_size: Dict[Compiled, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledASTRunner): kernargs_size[ji.prg.device] += round_up(ctypes.sizeof(ji.prg.clprg.args_struct_t), 16)
    kernargs_ptrs: Dict[Compiled, int] = {dev:dev.allocator._alloc(sz) for dev,sz in kernargs_size.items()}

    # Fill initial arguments.
    self.ji_kargs_structs: Dict[int, ctypes.Structure] = {}
    for j,ji in enumerate(self.jit_cache):
      if not isinstance(ji.prg, CompiledASTRunner): continue
      self.ji_kargs_structs[j] = ji.prg.clprg.args_struct_t.from_address(kernargs_ptrs[ji.prg.device])
      kernargs_ptrs[ji.prg.device] += round_up(ctypes.sizeof(ji.prg.clprg.args_struct_t), 16)
      for i in range(len(ji.rawbufs)): self.ji_kargs_structs[j].__setattr__(f'f{i}', cast(Buffer, ji.rawbufs[i])._buf)
      for i in range(len(ji.prg.vars)): self.ji_kargs_structs[j].__setattr__(f'v{i}', var_vals[ji.prg.vars[i]])

    # Build queues.
    self.virt_aql_queues: Dict[Compiled, VirtAQLQueue] = {dev:VirtAQLQueue(dev, 2*len(self.jit_cache)+16) for dev in self.devices}
    self.packets = {}
    self.transfers = []
    self.ji_to_transfer: Dict[int, int] = {} # faster to store transfers as list and update using this mapping table.
    self.signals_to_reset: List[hsa.hsa_signal_t] = []
    self.w_dependency_map: Dict[Any, Union[hsa.hsa_signal_t, int]] = {}
    self.r_dependency_map: Dict[Any, List[Union[hsa.hsa_signal_t, int]]] = collections.defaultdict(list)
    signals_to_devices: Dict[ctypes.c_uint64, List[HSADevice]] = {}
    self.profile_info: Dict[Compiled, List[Tuple[Any, ...]]] = collections.defaultdict(list)

    # Special packet to wait for the world.
    self.kickoff_signals: Dict[HSADevice, hsa.hsa_signal_t] = {}
    for dev in self.devices: self.kickoff_signals[dev] = self.virt_aql_queues[dev].submit_barrier(need_signal=True)
    self.signals_to_reset += list(self.kickoff_signals.values())

    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledASTRunner):
        wait_signals = self.access_resources(ji.rawbufs[(outs:=ji.prg.outcount):], ji.rawbufs[:outs], new_dependency=j, sync_with_aql_packets=False)
        for i in range(0, len(wait_signals), 5):
          self.virt_aql_queues[ji.prg.device].submit_barrier(wait_signals=wait_signals[i:i+5])
        self.packets[j] = hsa.hsa_kernel_dispatch_packet_t.from_address(self.virt_aql_queues[ji.prg.device].write_addr)
        sync_signal = self.virt_aql_queues[ji.prg.device].submit_kernel(ji.prg.clprg, *ji.prg.launch_dims(var_vals), #type:ignore
                                                                        ctypes.addressof(self.ji_kargs_structs[j]), need_signal=PROFILE)
        if PROFILE:
          self.profile_info[ji.prg.device].append((sync_signal, ji.prg.clprg.name, False))
          self.signals_to_reset.append(sync_signal)
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.rawbufs[0:2]]
        dest_dev, src_dev = cast(HSADevice, dest.d), cast(HSADevice, src.d)
        sync_signal = init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(x))))
        self.signals_to_reset.append(sync_signal)
        signals_to_devices[sync_signal.handle] = [dest_dev, src_dev]

        wait_signals = self.access_resources(read=[src], write=[dest], new_dependency=sync_signal, sync_with_aql_packets=True)
        self.transfers.append([dest._buf, dest_dev.agent, src._buf, src_dev.agent, dest.nbytes, len(wait_signals),
                              (hsa.hsa_signal_t*len(wait_signals))(*wait_signals), sync_signal, hsa.HSA_AMD_SDMA_ENGINE_0, True])
        self.ji_to_transfer[j] = len(self.transfers) - 1
        if PROFILE: self.profile_info[src_dev].append((sync_signal, f"transfer: HSA:{src_dev.device_id} -> HSA:{dest_dev.device_id}", True))

    # Wait for all active signals to finish the graph
    wait_signals_to_finish: Dict[HSADevice, List[hsa.hsa_signal_t]] = collections.defaultdict(list)
    for v in dedup_signals(list(self.w_dependency_map.values()) + list(itertools.chain.from_iterable(self.r_dependency_map.values()))):
      for dev in signals_to_devices[v.handle]:
        wait_signals_to_finish[dev].append(v)

    self.finish_signal = init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(x))))
    for dev in self.devices:
      wait_signals = wait_signals_to_finish[dev]
      for i in range(0, max(1, len(wait_signals)), 5):
        self.virt_aql_queues[dev].submit_barrier(wait_signals[i:i+5], need_signal=(i+5>=len(wait_signals)), completion_signal=self.finish_signal)

    # Zero signals to allow graph to start and execute.
    for sig in self.signals_to_reset: hsa.hsa_signal_silent_store_relaxed(sig, 0)
    hsa.hsa_signal_silent_store_relaxed(self.finish_signal, 0)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # Wait and restore signals
    hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    for sig in self.signals_to_reset: hsa.hsa_signal_silent_store_relaxed(sig, 1)
    hsa.hsa_signal_silent_store_relaxed(self.finish_signal, len(self.devices))

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items():
      if j in self.ji_kargs_structs:
        self.ji_kargs_structs[j].__setattr__(f'f{i}', input_rawbuffers[input_idx]._buf)
      else:
        if i == 0: self.transfers[self.ji_to_transfer[j]][0] = input_rawbuffers[input_idx]._buf # dest
        elif i == 1: self.transfers[self.ji_to_transfer[j]][2] = input_rawbuffers[input_idx]._buf # src

    # Update var_vals
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledASTRunner, self.jit_cache[j].prg).vars):
        self.ji_kargs_structs[j].__setattr__(f'v{i}', var_vals[v])

    # Update launch dims
    for j in self.jc_idxs_with_updatable_launch_dims:
      gl, lc = cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals)
      self.packets[j].workgroup_size_x = lc[0]
      self.packets[j].workgroup_size_y = lc[1]
      self.packets[j].workgroup_size_z = lc[2]
      self.packets[j].grid_size_x = gl[0] * lc[0]
      self.packets[j].grid_size_y = gl[1] * lc[1]
      self.packets[j].grid_size_z = gl[2] * lc[2]

    for dev in self.devices:
      dev.flush_hdp()
      dev.hw_queue.blit_packets(self.virt_aql_queues[dev].queue_base, self.virt_aql_queues[dev].packets_count)

    for transfer_data in self.transfers:
      check(hsa.hsa_amd_memory_async_copy_on_engine(*transfer_data))

    et = None
    if wait:
      st = time.perf_counter()
      hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      et = time.perf_counter() - st

    for profdev,profdata in self.profile_info.items(): Profiler.tracked_signals[profdev] += profdata
    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers),
                 jit=jit, num_kernels=len(self.jit_cache), device="HSA")
    return et

  def dependency_as_signal(self, dep, sync_with_aql_packets) -> Optional[hsa.hsa_signal_t]:
    if isinstance(dep, hsa.hsa_signal_t): return dep
    elif sync_with_aql_packets and isinstance(packet := self.packets.get(dep), hsa.hsa_kernel_dispatch_packet_t):
      if packet.completion_signal.handle == EMPTY_SIGNAL.handle:
        packet.completion_signal = init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(x))))
        self.signals_to_reset.append(packet.completion_signal)
      return packet.completion_signal
    return None

  def access_resources(self, read, write, new_dependency=None, sync_with_aql_packets=False):
    # To synchronize access to resources, we monitor the necessary prerequisites for accessing each resource,
    # whether for write or read operations. A resource can be accessed by either a single writer or multiple readers.
    # The tracked dependencies are either hsa signals or ints that reference a specific aql packet.
    wait_signals: List[Optional[hsa.hsa_signal_t]] = []

    if sync_with_aql_packets: wait_signals += [self.kickoff_signals[rawbuf.d] for rawbuf in read+write]
    for rawbuf in read:
      wait_signals.append(self.dependency_as_signal(self.w_dependency_map.get(rawbuf._buf), sync_with_aql_packets=sync_with_aql_packets))
    for rawbuf in write:
      wait_signals.append(self.dependency_as_signal(self.w_dependency_map.get(rawbuf._buf), sync_with_aql_packets=sync_with_aql_packets))
      if rawbuf._buf in self.r_dependency_map:
        rdeps = self.r_dependency_map.pop(rawbuf._buf)

        # When synchronizing to aql packets, we only need to sync to the latest one, as they are executed in order.
        signal_deps, aql_deps = [x for x in rdeps if isinstance(x, hsa.hsa_signal_t)], [x for x in rdeps if isinstance(x, int)]
        deps = signal_deps + [max(aql_deps)] if aql_deps else []
        for dep in deps: wait_signals.append(self.dependency_as_signal(dep, sync_with_aql_packets=sync_with_aql_packets))

    if new_dependency is not None:
      for rawbuf in read: self.r_dependency_map[rawbuf._buf].append(new_dependency)
      for rawbuf in write: self.w_dependency_map[rawbuf._buf] = new_dependency

    return dedup_signals(wait_signals)
