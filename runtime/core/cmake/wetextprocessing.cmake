FetchContent_Declare(wetextprocessing
  GIT_REPOSITORY https://github.com/wenet-e2e/WeTextProcessing.git
  GIT_TAG        origin/master
)
FetchContent_MakeAvailable(wetextprocessing)
include_directories(${wetextprocessing_SOURCE_DIR}/runtime )
add_subdirectory(${wetextprocessing_SOURCE_DIR}/runtime/utils)
add_subdirectory(${wetextprocessing_SOURCE_DIR}/runtime/processor)


