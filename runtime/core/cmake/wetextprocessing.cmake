FetchContent_Declare(wetextprocessing
  GIT_REPOSITORY https://github.com/wenet-e2e/WeTextProcessing.git
  GIT_TAG        origin/master
)
FetchContent_MakeAvailable(wetextprocessing)
include_directories(${wetextprocessing_SOURCE_DIR}/runtime )
## build independent wetext, whcih meets some issues:
# since wenet has utils as its subdirectory, 
# we must compile the utils in wetext into another lib name.
add_library(wetext_utils STATIC 
  ${wetextprocessing_SOURCE_DIR}/runtime/utils/string.cc )

# since wenet has utils/string.h which is the same name as in wetext, 
# the compiler find wenet::utils/string.h first. we should use wetext::utils/string.h
# 获取项目中所有包含文件夹的路径
get_property(includes DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# 遍历并输出每个包含文件夹的路径
foreach(include ${includes})
  message("Include directory: ${include}")
endforeach()
target_link_libraries(wetext_utils PUBLIC glog)

add_library(wetext_processor STATIC ${wetextprocessing_SOURCE_DIR}/runtime/utils/string.cc 
  ${wetextprocessing_SOURCE_DIR}/runtime/processor/processor.cc
  ${wetextprocessing_SOURCE_DIR}/runtime/processor/token_parser.cc)
if(MSVC)
  target_link_libraries(wetext_processor PUBLIC fst wetext_utils)
else()
  target_link_libraries(wetext_processor PUBLIC dl fst wetext_utils)
endif()
 

