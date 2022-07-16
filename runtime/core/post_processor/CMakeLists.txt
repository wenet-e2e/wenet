add_library(post_processor STATIC
  post_processor.cc
)
if(ITN)
  target_link_libraries(post_processor PUBLIC utils utils text_processor)
else()
  target_link_libraries(post_processor PUBLIC utils)
endif()
