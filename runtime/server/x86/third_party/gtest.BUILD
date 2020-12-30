cc_library(
    name = "gtest",
    srcs = glob(
        ["src/*.cc"],
        exclude = ["src/gtest-all.cc"]
    ),
    hdrs = glob([
        "include/**/*.h",
        "src/*.h"
    ]),
    includes = [
        "include",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
