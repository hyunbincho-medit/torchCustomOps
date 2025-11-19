import os
import platform
import sys

from setuptools import find_packages, setup
from torch.__config__ import parallel_info
from torch.utils import cpp_extension

__version__ = "0.1.0"
# WITH_CUDA = os.getenv("WITH_CUDA", "1") == "1"
WITH_CUDA = False

sources = [
    "csrc/fpsample.cpp",
    "csrc/fpsample_autograd.cpp",
    "csrc/fpsample_meta.cpp",
    "csrc/cpu/fpsample_cpu.cpp",
    "csrc/cpu/bucket_fps/wrapper.cpp",
]

# KNN sources
knn_sources = [
    "csrc/knn_torch3d.cpp",
    "csrc/knn/knn_cpu.cpp",
]

extra_compile_args = {"cxx": ["-O3"]}
extra_link_args = []

# OpenMP
info = parallel_info()
if "backend: OpenMP" in info and "OpenMP not found" not in info and sys.platform != "darwin":
    extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
    if sys.platform == "win32":
        extra_compile_args["cxx"] += ["/openmp"]
    else:
        extra_compile_args["cxx"] += ["-fopenmp"]
else:
    print("Compiling without OpenMP...")

# Compile for mac arm64
if sys.platform == "darwin":
    extra_compile_args["cxx"] += ["-D_LIBCPP_DISABLE_AVAILABILITY"]
    if platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]


if WITH_CUDA:
    # TODO
    raise NotImplementedError("CUDA is not supported yet.")
    sources += []
    ext_modules = [
        cpp_extension.CUDAExtension(
            name="torch_fpsample._core",
            include_dirs=["csrc"],
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
else:
    ext_modules = [
        cpp_extension.CppExtension(
            name="torch_fpsample._core",
            include_dirs=["csrc"],
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),

        cpp_extension.CppExtension(
            name="knn_torch3d._core",
            include_dirs=["csrc"],
            sources=knn_sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]


setup(
    name="torch_custom_ops",  # 이름 변경 (두 확장 모두 포함)
    version=__version__,
    author="Hyunbin cho",
    author_email="hyunbin.cho@medit.com",
    description="PyTorch custom operations including fpsample and KNN.",
    ext_modules=ext_modules,
    keywords=["pytorch", "farthest", "furthest", "sampling", "sample", "fps", "knn", "nearest", "neighbors"],
    packages=find_packages(),
    package_data={"": ["*.pyi"]},
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"]
)
