#!/bin/bash
set -e

usage() {
    echo "Usage:"
    echo "bash compile.sh [-r] [-d] [-c]"
    echo "Description:"
    echo "-r, build release."
    echo "-d, build debug."
    echo "-c, remove cmakecache or build dir, then build."
    echo "Example 1:"
    echo "  ./compile.sh -r "
    echo "  means: remove cache files in build dir, then build release."
    echo "Example 2:"
    echo "  ./compile.sh -d -c all "
    echo "  means: remove all files in build dir, then build debug."
    exit -1
}

if [ -z $CXX ]; then
  echo -e "\033[31m [WARNING]: NO CXX in your env. Suggest setting CXX variable to support C++14. \033[0m"
  sleep 2
fi

build_type='Release'
clean_type='cache'

while getopts 'rdc:h' OPT; do
    case $OPT in
        r) build_type="Release";;
        d) build_type="Debug";;
        c) clean_type="$OPTARG";;
        h) usage;;
        ?) usage;;
    esac
done

if [ ! -d ./build ];then
  mkdir build
fi

if [ "$clean_type" = "all" ];then
  pushd build
  rm -rf ./*
  popd
else
  pushd build
  rm -rf CMakeFiles/ cmake_install.cmake CMakeCache.txt CPackSourceConfig.cmake
  popd
fi

build_cmd="cd build && cmake -DINTTYPES_FORMAT:STRING=C99 "

if [ "$build_type" = "Release" ];then
  build_cmd="${build_cmd} -DCMAKE_BUILD_TYPE=Release .. && cmake --build ./ "
else
  build_cmd="${build_cmd} -DCMAKE_BUILD_TYPE=Debug .. && cmake --build ./ "
fi

echo "build command is ${build_cmd}"

eval ${build_cmd}
