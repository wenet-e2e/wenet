
cc_binary(
    name = "decoder_main",
    srcs = ["decoder_main.cc"],
    deps = [
        "//utils:utils",
        "//decoder:decoder",
        "//frontend:frontend",
        "@com_github_google_glog//:glog",
        "@com_github_gflags_gflags//:gflags",
        "@libtorch//:torch",
    ],
)

cc_binary(
    name = "websocket_client_main",
    srcs = ["websocket_client_main.cc"],
    deps = [
        "//frontend:frontend",
        "//websocket:websocket_client",
        "@com_github_gflags_gflags//:gflags",
    ],
)

cc_binary(
    name = "websocket_server_main",
    srcs = ["websocket_server_main.cc"],
    deps = [
        "//websocket:websocket_server",
        "@com_github_gflags_gflags//:gflags",
    ],
)
