FetchContent_Declare(googletest
  URL      https://github.com/google/googletest/archive/release-1.11.0.zip
  URL_HASH SHA256=353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a
)
if(MSVC)
  set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
endif()
FetchContent_MakeAvailable(googletest)