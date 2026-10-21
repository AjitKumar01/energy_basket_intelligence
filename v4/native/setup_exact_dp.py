"""Build the exact_dp PyTorch extension (called by v4/kernels.py; no Ninja needed)."""
import platform
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = Path(__file__).resolve().parent
if platform.system() == "Darwin" and platform.machine() == "arm64":
    FLAGS = ["-O3", "-mcpu=native"]
else:
    FLAGS = ["-O3", "-march=native"]

setup(name="v4-exact-dp",
      ext_modules=[CppExtension("v4_exact_dp", [str(HERE / "exact_dp.cpp")],
                                extra_compile_args=FLAGS)],
      cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)})
