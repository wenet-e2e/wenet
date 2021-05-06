cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(kaldi)

add_library(kaldi
base/kaldi-math.cc
base/kaldi-error.cc
decoder/lattice-faster-decoder.cc
decoder/lattice-faster-online-decoder.cc
lat/determinize-lattice-pruned.cc
lat/lattice-functions.cc
lm/arpa-file-parser.cc
lm/arpa-lm-compiler.cc
util/text-utils.cc
util/parse-options.cc
)

target_link_libraries(kaldi PUBLIC utils)

add_executable(arpa2fst lmbin/arpa2fst.cc)
target_link_libraries(arpa2fst PUBLIC kaldi)
