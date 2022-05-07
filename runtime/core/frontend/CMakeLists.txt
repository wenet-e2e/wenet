add_library(frontend STATIC
  feature_pipeline.cc
  fft.cc
)
target_link_libraries(frontend PUBLIC utils)