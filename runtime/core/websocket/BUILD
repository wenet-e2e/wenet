package(default_visibility = ["//visibility:public"])

cc_library(
    name = 'websocket_server',
    srcs = ['websocket_server.cc'],
    hdrs = ['websocket_server.h'],
    deps = [
        "//decoder:decoder",
        "@boost//:boost",
    ],
)

cc_library(
    name = 'websocket_client',
    srcs = ['websocket_client.cc'],
    hdrs = ['websocket_client.h'],
    deps = [
        "//utils:utils",
        "@boost//:boost",
        "@com_github_google_glog//:glog",
    ],
)

