cmake_minimum_required(VERSION 3.16)
project(main-wasm-module)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED EMSCRIPTEN_MALLOC)
    set(EMSCRIPTEN_MALLOC dlmalloc)
endif()
# Do not use ALLOW_MEMORY_GROWTH until https://github.com/emscripten-core/emscripten/issues/24287 is enabled, right
# now is too error-prone.
if(NOT DEFINED EMSCRIPTEN_MEMORY_MB)
    set(EMSCRIPTEN_MEMORY_MB 64)
endif()
math(EXPR EMSCRIPTEN_MEMORY "${EMSCRIPTEN_MEMORY_MB} * 1024 * 1024")

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT EMSCRIPTEN_MALLOC STREQUAL "mimalloc")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

if(EMSCRIPTEN)
    include_directories(
        3rdparty/kiss_fft
        3rdparty/pffft
    )
    add_executable(main-wasm-module
        3rdparty/kiss_fft/kiss_fft.c
        3rdparty/kiss_fft/kiss_fftr.c
        3rdparty/pffft/pffft.c
        src/main.cpp
    )
    target_compile_options(main-wasm-module PRIVATE
        -msimd128
        -Wall
        -fstack-protector
        -ftrivial-auto-var-init=zero
        -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST
    )
    target_link_options(main-wasm-module PRIVATE
        -lembind
        -sSTRICT
        -sRETAIN_COMPILER_SETTINGS
        -sMALLOC=${EMSCRIPTEN_MALLOC}
        -sINITIAL_MEMORY=${EMSCRIPTEN_MEMORY}
        -sMODULARIZE
        -sENVIRONMENT=[web,webview,worker]
        -sEXPORT_ES6
        -sEXPORT_NAME='createMainModule'
        -sEXPORTED_RUNTIME_METHODS=[HEAPF32]
        --no-entry
        --emit-tsd main-wasm-module.d.ts
        --emit-symbol-map
    )
else()
    message(FATAL_ERROR "This project must be built with Emscripten. Please use 'emcmake cmake' instead of 'cmake'.")
endif()
