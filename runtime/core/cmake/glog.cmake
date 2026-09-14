# Disable dbghelp (Windows only) for non-Windows platforms
set(WITH_DBGHELP OFF CACHE BOOL "Disable dbghelp on non-Windows platforms" FORCE)

FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

# Remove dbghelp from glog interface link libraries on non-Windows platforms
# glog incorrectly adds dbghelp as a link dependency even on non-Windows
if(NOT WIN32)
  get_target_property(glog_interface_libs glog INTERFACE_LINK_LIBRARIES)
  if(glog_interface_libs)
    set(new_libs "")
    foreach(lib ${glog_interface_libs})
      string(FIND "${lib}" "dbghelp" dbghelp_pos)
      if(dbghelp_pos EQUAL -1)
        list(APPEND new_libs "${lib}")
      endif()
    endforeach()
    set_target_properties(glog PROPERTIES INTERFACE_LINK_LIBRARIES "${new_libs}")
  endif()
endif()
