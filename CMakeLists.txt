cmake_minimum_required(VERSION 3.16)
project(plt2 CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Threads REQUIRED)

# SIMD path is picked from the target ISA (identical stream on all of them):
#   AVX-512BW -> avx512 | AVX2 -> avx2 | SSE4.1 (Nehalem..Ivy Bridge) -> sse4.1 | older x86 -> bundled SIMDe (avx2.h)
# PLT_ARCH: empty = "native" for gcc/clang/clang-cl, /QxHost for ICX; for cl.exe (no "native") set AVX, AVX2 or AVX512.
#           Any other value is passed as -march=<value> (e.g. ivybridge, haswell, x86-64-v2) for gcc/clang/clang-cl.
set(PLT_ARCH "" CACHE STRING "target ISA: empty = native; cl.exe: AVX|AVX2|AVX512; gcc/clang: -march value")
option(PLT_SCALAR "force the scalar path" OFF)

add_executable(plt2 plt2.cpp plt2_tables.h avx2.h)
target_link_libraries(plt2 PRIVATE Threads::Threads)
if(PLT_SCALAR)
  target_compile_definitions(plt2 PRIVATE PLT_SCALAR)
endif()

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")                       # icx
    target_compile_options(plt2 PRIVATE /O3 $<IF:$<BOOL:${PLT_ARCH}>,/arch:${PLT_ARCH},/QxHost>)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")                       # clang-cl
    target_compile_options(plt2 PRIVATE /O2 -march=$<IF:$<BOOL:${PLT_ARCH}>,${PLT_ARCH},native>)
  else()                                                               # cl.exe: no -march, no native
    target_compile_options(plt2 PRIVATE /O2 /Ob3 /Zc:preprocessor)
    if(PLT_ARCH)
      target_compile_options(plt2 PRIVATE /arch:${PLT_ARCH})
    endif()
  endif()
else()                                                                 # gcc / clang
  target_compile_options(plt2 PRIVATE -O3 -march=$<IF:$<BOOL:${PLT_ARCH}>,${PLT_ARCH},native>)
endif()

# PALIC-1 (llrans, p-conv) from palic-1/; executables land next to plt2
option(PLT_BUILD_PALIC1 "build palic-1 (llrans, p-conv)" ON)
if(PLT_BUILD_PALIC1)
  add_subdirectory(palic-1)
  set_target_properties(llrans p-conv PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR} RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
endif()

# pctx (PALIC + activity contexts, FSE/Huff0 streams) from palic-ctx/
option(PLT_BUILD_PCTX "build palic-ctx (pctx)" ON)
if(PLT_BUILD_PCTX)
  enable_language(C)
  add_subdirectory(palic-ctx)
  set_target_properties(pctx PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR} RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
endif()

# hlc (HALIC-class on PALIC-1: one RCT, independent 256x256 tiles, in-place decode) from hlc/
option(PLT_BUILD_HLC "build hlc" ON)
if(PLT_BUILD_HLC AND PLT_BUILD_PCTX)
  add_subdirectory(hlc hlc_build)              # binary dir must not clash with the hlc executable
  set_target_properties(hlc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR} RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
endif()

# ctx4d research kernels (HLC5 4-D context: AVX2 encoder kernel + wavefront decoder, SIMDe fallback) from hlc/research/ctx4d
option(PLT_BUILD_CTX4D "build the ctx4d research kernels (ctx4d_enc, ctx4d_dec)" ON)
if(PLT_BUILD_CTX4D)
  add_subdirectory(hlc/research/ctx4d ctx4d_build)
endif()
