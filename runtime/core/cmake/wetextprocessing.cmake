FetchContent_Declare(wetextprocessing
  GIT_REPOSITORY https://github.com/wenet-e2e/WeTextProcessing.git
  GIT_TAG        origin/main
)
FetchContent_MakeAvailable(wetextprocessing)
include_directories(${wetextprocessing_SOURCE_DIR}/src ${wetextprocessing_BINARY_DIR})
add_subdirectory(${wetextprocessing_SOURCE_DIR}/src ${wetextprocessing_BINARY_DIR})
add_definitions(-DUSE_ITN)
