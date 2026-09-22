"""HIPCC helper for custom kernels that select a ROCm toolchain per compilation."""
import hashlib, os, pathlib, shutil, subprocess, tempfile
from tinygrad.device import Compiler
from tinygrad.helpers import amdgpu_disassemble, getenv

class HIPCCCompiler(Compiler):
  def __init__(self, arch:str, extra_options:list[str]=[], *, hipcc_path:str|pathlib.Path|None=None,
               rocm_path:str|pathlib.Path|None=None):
    self.arch, self.extra_options, self.no_hipcc = arch, extra_options, getenv("NO_HIPCC")
    hipcc_cmd = str(hipcc_path) if hipcc_path is not None else "hipcc"
    self.hipcc_path = str(pathlib.Path(shutil.which(hipcc_cmd) or hipcc_cmd).resolve())
    self.rocm_path = str(pathlib.Path(rocm_path if rocm_path is not None else getenv("ROCM_PATH", "/opt/rocm")).resolve())
    toolchain_tag = hashlib.sha256(f"{self.hipcc_path}\0{self.rocm_path}".encode()).hexdigest()[:8]
    super().__init__(f"compile_hipcc_{self.arch}_{hashlib.sha256(' '.join(extra_options).encode()).hexdigest()[:8]}"+
                     f"_tc{toolchain_tag}"+("_nohipcc" if self.no_hipcc else ""))
  def compile(self, src:str) -> bytes:
    if self.no_hipcc: return b""
    compiler_env = {**os.environ, "PATH": f"{pathlib.Path(self.hipcc_path).parent}{os.pathsep}{os.environ.get('PATH', '')}",
                    "ROCM_PATH": self.rocm_path}
    with tempfile.NamedTemporaryFile(suffix=".cpp") as srcf, tempfile.NamedTemporaryFile(suffix=".bc") as bcf:
      with tempfile.NamedTemporaryFile(suffix=".hsaco") as libf:
        srcf.write(src.encode())
        srcf.flush()

        subprocess.run([self.hipcc_path, "-c", "-emit-llvm", "--cuda-device-only", "-O3", "-mcumode",
                        f"--offload-arch={self.arch}", f"-I{self.rocm_path}/include/hip", "-o", bcf.name, srcf.name] + self.extra_options,
                       check=True, env=compiler_env)
        subprocess.run([self.hipcc_path, "-target", "amdgcn-amd-amdhsa", f"-mcpu={self.arch}",
                        "-O3", "-mllvm", "-amdgpu-internalize-symbols", "-c", "-o", libf.name, bcf.name] + self.extra_options,
                       check=True, env=compiler_env)

        return pathlib.Path(libf.name).read_bytes()
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)
