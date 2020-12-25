package(default_visibility = ["//visibility:public"])

cc_library(
    name = 'ctc_prefix_beam_search',
    srcs = ['ctc_prefix_beam_search.cc'],
    hdrs = ['ctc_prefix_beam_search.h'],
    deps = [
		"@com_github_google_glog//:glog",
        "@libtorch//:torch",
    ],
)

cc_test(
    name = 'ctc_prefix_beam_search_test',
    srcs = ['ctc_prefix_beam_search_test.cc'],
    deps = [
        ":ctc_prefix_beam_search",
        "@com_github_google_gtest//:gtest",
    ],
)

cc_library(
    name = 'torch_asr_model',
    srcs = ['torch_asr_model.cc'],
    hdrs = ['torch_asr_model.h'],
    deps = [
		"@com_github_google_glog//:glog",
        "@libtorch//:torch",
    ],
)

cc_library(
    name = 'torch_asr_decoder',
    srcs = ['torch_asr_decoder.cc'],
    hdrs = ['torch_asr_decoder.h', 'symbol_table.h'],
    deps = [
		"@com_github_google_glog//:glog",
        "@libtorch//:torch",
        "//frontend:frontend",
        ":torch_asr_model",
        ":ctc_prefix_beam_search",
    ],
)

cc_library(
    name = 'decoder',
    deps = [
        ":torch_asr_decoder",
    ],
)
