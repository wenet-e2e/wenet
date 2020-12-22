licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "torch",
    srcs = [
        "lib/libtorch.so",
        "lib/libc10.so",
        "lib/libtorch_cpu.so",
        "lib/libgomp-75eea7e8.so.1",
        "lib/libtensorpipe.so",
    ],
    hdrs = glob(["include/**/*.h"]),
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    copts = ["-D_GLIBCXX_USE_CXX11_ABI=1"],
    visibility = ["//visibility:public"]
)
