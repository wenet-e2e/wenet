cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(kaldi)

# include_directories() is called in the root CMakeLists.txt

add_library(kaldi-util
  base/kaldi-error.cc
  base/kaldi-math.cc
  util/kaldi-io.cc
  util/parse-options.cc
  util/simple-io-funcs.cc
  util/text-utils.cc
)
target_link_libraries(kaldi-util PUBLIC utils)

add_library(kaldi-decoder
  lat/determinize-lattice-pruned.cc
  lat/lattice-functions.cc
  decoder/lattice-faster-decoder.cc
  decoder/lattice-faster-online-decoder.cc
)
target_link_libraries(kaldi-decoder PUBLIC kaldi-util)

if(GRAPH_TOOLS)
  # Arpa binary
  add_executable(arpa2fst
    lm/arpa-file-parser.cc
    lm/arpa-lm-compiler.cc
    lmbin/arpa2fst.cc
  )
  target_link_libraries(arpa2fst PUBLIC kaldi-util)

  # FST tools binary
  set(FST_BINS
    fstaddselfloops
    fstdeterminizestar
    fstisstochastic
    fstminimizeencoded
    fsttablecompose
  )

  if(NOT MSVC)
    # dl is for dynamic linking, otherwise there is a linking error on linux
    link_libraries(dl)
  endif()
  foreach(name IN LISTS FST_BINS)
    add_executable(${name}
      fstbin/${name}.cc
      fstext/kaldi-fst-io.cc
    )
    target_link_libraries(${name} PUBLIC kaldi-util)
  endforeach()
endif()
