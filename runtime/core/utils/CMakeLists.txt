add_library(utils STATIC
  string.cc
  utils.cc
)

if(NOT ANDROID)
  if(MSVC)
    target_link_libraries(utils PUBLIC fst)
  else()
    target_link_libraries(utils PUBLIC fst dl)
  endif()
endif()