cmake_minimum_required(VERSION 3.20)
project(diffusion_cpp_rs LANGUAGES C CXX)

# Build only the library. Media encoding belongs to the Rust application.
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_STATIC OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
# Optional CUTLASS kernels require a large extra submodule. Users can opt in
# with -DSD_CUDA_CUTLASS_PATH=/path/to/cutlass through CMAKE_ARGS.
set(SD_CUDA_CUTLASS_PATH "" CACHE PATH "Optional CUTLASS source tree")
set(GGML_METAL ${SD_METAL} CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

# GGML otherwise enables AVX2 when GGML_NATIVE is off. Keep the default x86
# build portable; explicit CMAKE_ARGS can still select a deployment baseline.
if(NOT GGML_NATIVE)
    foreach(feature SSE42 AVX AVX2 BMI2 FMA F16C)
        set(GGML_${feature} OFF CACHE BOOL "")
    endforeach()
endif()

add_subdirectory("${SD_CPP_SOURCE_DIR}" stable-diffusion)

# Use CMake's target paths instead of guessing archive locations across build
# generators, platforms, configurations and optional GPU backends.
set(RUST_LINK_LIBRARIES "")
foreach(target stable-diffusion ggml ggml-cpu ggml-metal ggml-cuda ggml-vulkan ggml-base)
    if(TARGET ${target})
        string(APPEND RUST_LINK_LIBRARIES "static|$<TARGET_LINKER_FILE:${target}>\n")
    endif()
endforeach()
if(SD_CUDA)
    foreach(target CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
        string(APPEND RUST_LINK_LIBRARIES "dylib|$<TARGET_LINKER_FILE:${target}>\n")
    endforeach()
endif()
if(SD_VULKAN)
    string(APPEND RUST_LINK_LIBRARIES "dylib|$<TARGET_LINKER_FILE:Vulkan::Vulkan>\n")
endif()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/native-libraries-$<CONFIG>.txt" CONTENT "${RUST_LINK_LIBRARIES}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/native-libraries-$<CONFIG>.txt" DESTINATION . RENAME native-libraries.txt)
