FetchContent_Declare(portaudio
  URL      https://github.com/PortAudio/portaudio/archive/refs/tags/v19.7.0.tar.gz
  URL_HASH SHA256=5af29ba58bbdbb7bbcefaaecc77ec8fc413f0db6f4c4e286c40c3e1b83174fa0
)
FetchContent_MakeAvailable(portaudio)
include_directories(${portaudio_SOURCE_DIR}/include)
