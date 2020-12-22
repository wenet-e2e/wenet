cc_library(
    name = 'ctc_prefix_beam_search',
    srcs = ['ctc_prefix_beam_search.cc'],
    hdrs = ['ctc_prefix_beam_search.h'],
    deps = [
		"@com_github_google_glog//:glog",
        "@com_github_gflags_gflags//:gflags",
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
