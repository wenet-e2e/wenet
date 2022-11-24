#!/usr/bin/env python3
# Copyright (c)  2020  Xiaomi Corporation (author: Fangjun Kuang)
#                2022  Binbin Zhang(binbzha@qq.com)

import glob
import os
import shutil
import sys

import setuptools
from setuptools.command.build_ext import build_ext


def cmake_extension(name, *args, **kwargs) -> setuptools.Extension:
    kwargs["language"] = "c++"
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext):
    def build_extension(self, ext: setuptools.extension.Extension):
        os.makedirs(self.build_temp, exist_ok=True)
        os.makedirs(self.build_lib, exist_ok=True)

        cmake_args = os.environ.get("WENET_CMAKE_ARGS",
                                    "-DCMAKE_BUILD_TYPE=Release")
        if "PYTHON_EXECUTABLE" not in cmake_args:
            print(f"Setting PYTHON_EXECUTABLE to {sys.executable}")
            cmake_args += f" -DPYTHON_EXECUTABLE={sys.executable}"

        src_dir = os.path.dirname(os.path.abspath(__file__))
        os.system(f"cmake {cmake_args} -B {self.build_temp} -S {src_dir}")
        ret = os.system(f"""
            cmake --build {self.build_temp} --target _wenet --config Release
        """)
        if ret != 0:
            raise Exception(
                "\nBuild wenet failed. Please check the error message.\n"
                "You can ask for help by creating an issue on GitHub.\n"
                "\nClick:\n    https://github.com/wenet-e2e/wenet/issues/new\n"
            )

        libs = []
        for ext in ['so', 'pyd']:
            libs.extend(
                glob.glob(f"{self.build_temp}/**/_wenet*.{ext}",
                          recursive=True))
        for ext in ['so', 'dylib', 'dll']:
            libs.extend(
                glob.glob(f"{self.build_temp}/**/*wenet_api.{ext}",
                          recursive=True))

        for lib in libs:
            print(f"Copying {lib} to {self.build_lib}/")
            shutil.copy(f"{lib}", f"{self.build_lib}/")


def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme


package_name = "wenetruntime"

setuptools.setup(
    name=package_name,
    version='1.0.12',
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
    setup_requires=["tqdm"],
    install_requires=["torch", "tqdm"],
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache licensed, as found in the LICENSE file",
)
