add_library(http STATIC
  http_client.cc
  http_server.cc
)
target_link_libraries(http PUBLIC decoder)
