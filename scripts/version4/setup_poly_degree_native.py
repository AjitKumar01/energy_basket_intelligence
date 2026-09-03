"""Build the audit-only degree-aware C++ extension without Ninja."""
import platform
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent
if platform.system() == "Windows":
    OPTIMIZATION_FLAGS = ["/O2"]
elif platform.system() == "Darwin" and platform.machine() == "arm64":
    OPTIMIZATION_FLAGS = ["-O3", "-mcpu=native"]
else:
    OPTIMIZATION_FLAGS = ["-O3", "-march=native"]

setup(
    name="v3-poly-degree-native",
    ext_modules=[CppExtension(
        "v3_poly_degree_native",
        [str(HERE / "poly_degree_native.cpp")],
        extra_compile_args=OPTIMIZATION_FLAGS,
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
