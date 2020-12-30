

cc_binary(
    name = "decoder_main",
    srcs = ["decoder_main.cc"],
    deps = [
        "@com_github_google_glog//:glog",
        "@com_github_gflags_gflags//:gflags",
        "@libtorch//:torch",
        "//frontend:frontend",
        "//decoder:decoder",
    ],
)

