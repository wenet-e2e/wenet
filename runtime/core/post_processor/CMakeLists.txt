add_library(post_processor STATIC
  post_processor.cc
)
target_link_libraries(post_processor PUBLIC utils wetext_processor wetext_utils)

