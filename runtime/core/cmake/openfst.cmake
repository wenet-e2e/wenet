if(NOT ANDROID)
  include(gflags)
  # We can't build glog with gflags, unless gflags is pre-installed.
  # If build glog with pre-installed gflags, there will be conflict.
  set(WITH_GFLAGS OFF CACHE BOOL "whether build glog with gflags" FORCE)
  include(glog)

  if(NOT GRAPH_TOOLS)
    set(HAVE_BIN OFF CACHE BOOL "Build the fst binaries" FORCE)
    set(HAVE_SCRIPT OFF CACHE BOOL "Build the fstscript" FORCE)
  endif()
  set(HAVE_COMPACT OFF CACHE BOOL "Build compact" FORCE)
  set(HAVE_CONST OFF CACHE BOOL "Build const" FORCE)
  set(HAVE_GRM OFF CACHE BOOL "Build grm" FORCE)
  set(HAVE_FAR OFF CACHE BOOL "Build far" FORCE)
  set(HAVE_PDT OFF CACHE BOOL "Build pdt" FORCE)
  set(HAVE_MPDT OFF CACHE BOOL "Build mpdt" FORCE)
  set(HAVE_LINEAR OFF CACHE BOOL "Build linear" FORCE)
  set(HAVE_LOOKAHEAD OFF CACHE BOOL "Build lookahead" FORCE)
  set(HAVE_NGRAM OFF CACHE BOOL "Build ngram" FORCE)
  set(HAVE_SPECIAL OFF CACHE BOOL "Build special" FORCE)

  if(MSVC)
    add_compile_options(/W0 /wd4244 /wd4267)
  endif()

  # "OpenFST port for Windows" builds openfst with cmake for multiple platforms.
  # Openfst is compiled with glog/gflags to avoid log and flag conflicts with log and flags in wenet/libtorch.
  # To build openfst with gflags and glog, we comment out some vars of {flags, log}.h and flags.cc.
  set(openfst_SOURCE_DIR ${fc_base}/openfst-src CACHE PATH "OpenFST source directory")
    FetchContent_Declare(openfst
      URL           https://github.com/kkm000/openfst/archive/refs/tags/win/1.7.2.1.tar.gz
      URL_HASH      SHA256=e04e1dabcecf3a687ace699ccb43a8a27da385777a56e69da6e103344cc66bca
      #URL           https://github.com/kkm000/openfst/archive/refs/tags/win/1.6.5.1.tar.gz
      #URL_HASH      SHA256=02c49b559c3976a536876063369efc0e41ab374be1035918036474343877046e
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(openfst)
    add_dependencies(fst gflags glog)
    target_link_libraries(fst PUBLIC gflags_nothreads_static glog)
  include_directories(${openfst_SOURCE_DIR}/src/include)
else()
  set(openfst_BINARY_DIR ${build_DIR}/wenet-openfst-android-1.0.2.aar/jni)
  include_directories(${openfst_BINARY_DIR}/include)
  link_directories(${openfst_BINARY_DIR}/${ANDROID_ABI})
  link_libraries(log gflags_nothreads glog fst)
endif()
