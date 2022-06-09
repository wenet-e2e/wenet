add_library(websocket STATIC
  websocket_client.cc
  websocket_server.cc
)
target_link_libraries(websocket PUBLIC decoder)
