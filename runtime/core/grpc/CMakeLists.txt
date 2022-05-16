# compile wenet.proto
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/grpc")
add_custom_command(
  OUTPUT  ${PROTO_DIR}/wenet.pb.cc
          ${PROTO_DIR}/wenet.pb.h
          ${PROTO_DIR}/wenet.grpc.pb.cc
          ${PROTO_DIR}/wenet.grpc.pb.h
  COMMAND ${protobuf_BINARY_DIR}/protoc
  ARGS --grpc_out "${PROTO_DIR}"
    --cpp_out "${PROTO_DIR}"
    -I "${PROTO_DIR}"
    --plugin=protoc-gen-grpc=${grpc_BINARY_DIR}/grpc_cpp_plugin
    wenet.proto)

# grpc_server/client
link_directories(${protobuf_BINARY_DIR}/lib)
add_library(wenet_grpc STATIC
  grpc_client.cc
  grpc_server.cc
  wenet.pb.cc
  wenet.grpc.pb.cc
)
target_link_libraries(wenet_grpc PUBLIC grpc++ grpc++_reflection decoder frontend)