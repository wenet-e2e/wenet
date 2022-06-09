add_executable(decoder_main decoder_main.cc)
target_link_libraries(decoder_main PUBLIC decoder)

add_executable(websocket_client_main websocket_client_main.cc)
target_link_libraries(websocket_client_main PUBLIC websocket)

add_executable(websocket_server_main websocket_server_main.cc)
target_link_libraries(websocket_server_main PUBLIC websocket)

add_executable(label_checker_main label_checker_main.cc)
target_link_libraries(label_checker_main PUBLIC decoder)

if(TORCH)
 add_executable(api_main api_main.cc)
 target_link_libraries(api_main PUBLIC wenet_api)
endif()

if(GRPC)
  add_executable(grpc_server_main grpc_server_main.cc)
  target_link_libraries(grpc_server_main PUBLIC wenet_grpc)
  add_executable(grpc_client_main grpc_client_main.cc)
  target_link_libraries(grpc_client_main PUBLIC wenet_grpc)
endif()
