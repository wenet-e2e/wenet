def wenet_load_libs(backend="torchScript"):
    import platform
    import sys
    from ctypes import cdll
    from os import path

    root_path = path.dirname(__file__)
    libs_dirname = path.join(root_path, 'lib')


    # default for linux
    system = platform.system()
    if system == "Windows":
        suffix = ".dll"
    elif system == "Linux":
        suffix = ".so"
    else:
        sys.exit("only windowns and linux support for now")

    if backend == "torchScript":
        libs = [
            "libtorch{}",
            "libtorch_cpu{}",
            "libc10{}",
            "libfst{}.8",
            "libgomp-a34b3233{}.1",
        ]
    else:
        sys.exit("only torchScript model support for now");

    for lib in libs:
        cdll.LoadLibrary(path.join(libs_dirname, lib.format(suffix)))

wenet_load_libs()

from wenet.label_checker import LabelChecker
from wenet.lib._pywrap_wenet import Params
from wenet.model import StreammingAsrDecoder, WenetWrapper, load_model
from wenet.streamming_decoder import StreammingAsrDecoder
