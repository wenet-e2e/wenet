add_library(wenet_api SHARED wenet_api.cc)
target_link_libraries(wenet_api PUBLIC decoder frontend)
