package(default_visibility = ["//visibility:public"])

cc_library(
    name = 'frontend',
    hdrs = ['wav.h'],
    deps = [
        ':feature_pipeline',
    ],
)

cc_library(
    name = 'fft',
    srcs = ['fft.cc'],
    hdrs = ['fft.h'],
)

cc_library(
    name = 'feature_pipeline',
    srcs = ['feature_pipeline.cc'],
    hdrs = [
        'fbank.h',
        'feature_pipeline.h'
    ],
    deps = [
        ":fft",
        "//utils:utils",
        "@com_github_google_glog//:glog",
    ],
)
