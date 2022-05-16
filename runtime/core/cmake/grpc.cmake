include_directories(${CMAKE_CURRENT_SOURCE_DIR}/grpc)
# third_party: grpc
# On how to build grpc, you may refer to https://github.com/grpc/grpc
# We recommend manually recursive clone the repo to avoid internet connection problem
FetchContent_Declare(gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.37.1
)
FetchContent_MakeAvailable(gRPC)