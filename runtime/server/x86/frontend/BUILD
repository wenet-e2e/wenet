cc_library(
    name = 'fft',
    srcs = ['fft.cc'],
    hdrs = ['fft.h'],
)

cc_library(
    name = 'feature_pipeline',
    srcs = ['feature_pipeline.cc'],
    hdrs = ['fbank.h', 'feature_pipeline.h'],
    deps = [
        ":fft",
        "@com_github_google_glog//:glog",
    ],
)
