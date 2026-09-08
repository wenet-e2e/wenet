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

  set(openfst_SOURCE_DIR ${fc_base}/openfst-src CACHE PATH "OpenFST source directory")
  FetchContent_Declare(openfst
    URL      https://github.com/csukuangfj/openfst/archive/refs/tags/v1.8.5-2026-04-11.tar.gz
    URL_HASH SHA256=57fbc4b950ae81b1a0e1e298af15652da968a6723a592b7874e9b4027a80a5b4
  )
  FetchContent_MakeAvailable(openfst)
  include_directories(${openfst_SOURCE_DIR}/src/include)
else()
  set(openfst_BINARY_DIR ${build_DIR}/wenet-openfst-android-1.0.2.aar/jni)
  include_directories(${openfst_BINARY_DIR}/include)
  link_directories(${openfst_BINARY_DIR}/${ANDROID_ABI})
  link_libraries(log gflags_nothreads glog fst)
endif()
