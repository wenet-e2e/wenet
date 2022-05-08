#!/usr/bin/env python3
# Copyright (c)  2020  Xiaomi Corporation (author: Fangjun Kuang)
#                2022  Binbin Zhang(binbzha@qq.com)

import glob
import os
import platform
import shutil
import sys

import setuptools
from setuptools.command.build_ext import build_ext

cur_dir = os.path.dirname(os.path.abspath(__file__))


def is_windows():
    return platform.system() == "Windows"


def cmake_extension(name, *args, **kwargs) -> setuptools.Extension:
    kwargs["language"] = "c++"
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext):
    def build_extension(self, ext: setuptools.extension.Extension):
        build_dir = self.build_temp
        os.makedirs(build_dir, exist_ok=True)

        os.makedirs(self.build_lib, exist_ok=True)

        cmake_args = os.environ.get("WENET_CMAKE_ARGS", "")
        make_args = os.environ.get("WENET_MAKE_ARGS", "")
        system_make_args = os.environ.get("MAKEFLAGS", "")

        if cmake_args == "":
            cmake_args = "-DCMAKE_BUILD_TYPE=Release"

        if make_args == "" and system_make_args == "":
            print("For fast compilation, run:")
            print('export WENET_MAKE_ARGS="-j";')

        if "PYTHON_EXECUTABLE" not in cmake_args:
            print(f"Setting PYTHON_EXECUTABLE to {sys.executable}")
            cmake_args += f" -DPYTHON_EXECUTABLE={sys.executable}"

        if not is_windows():
            ret = os.system(f"""cd {build_dir};
                                cmake {cmake_args} {cur_dir};
                                cmake --build . --target _wenet
                             """)
            if ret != 0:
                raise Exception(
                    "\nBuild wenet failed. Please check the error message.\n"
                    "You can ask for help by creating an issue on GitHub.\n"
                    "\nClick:\n    https://github.com/wenet-e2e/wenet/issues/new\n"
                )

            lib_so = glob.glob(f"{build_dir}/**/*.so*", recursive=True)
            fst_lib = 'fc_base/openfst-subbuild/openfst-populate-prefix/lib'
            torch_lib = 'fc_base/libtorch-src/lib'
            lib_so.extend([
                f'{cur_dir}/{fst_lib}/libfst.so',
                f'{cur_dir}/{fst_lib}/libfstscript.so',
                f'{cur_dir}/{torch_lib}/libtorch.so',
                f'{cur_dir}/{torch_lib}/libtorch_cpu.so',
                f'{cur_dir}/{torch_lib}/libc10.so',
                f'{cur_dir}/{torch_lib}/libgomp-a34b3233.so.1',
            ])
            for so in lib_so:
                print(f"Copying {so} to {self.build_lib}/")
                shutil.copy(f"{so}", f"{self.build_lib}/")

            # macos
            # also need to copy *fst*.dylib
            # lib_so = glob.glob(f"{build_dir}/**/*.dylib*", recursive=True)
            # lib_so += glob.glob(f"{cur_dir}/fc_base/**/*.dylib*", recursive=True)
            # for so in lib_so:
            #     print(f"Copying {so} to {self.build_lib}/")
            #     shutil.copy(f"{so}", f"{self.build_lib}/", follow_symlinks=False)
        # for windows
        else:
            print('Windows is not supported')


def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme


package_name = "wenet"

setuptools.setup(
    name=package_name,
    version='1.0.1',
    author="Binbin Zhang",
    author_email="binbzha@qq.com",
    package_dir={
        package_name: "py",
    },
    packages=[package_name],
    url="https://github.com/wenet-e2e/wenet",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=[cmake_extension("_wenet")],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache licensed, as found in the LICENSE file",
    python_requires=">=3.6",
)
