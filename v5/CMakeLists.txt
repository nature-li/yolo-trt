cmake_minimum_required(VERSION 3.18)
project(yolo_server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Drogon REQUIRED)        # ← 新增

set(CUDA_ROOT /usr/local/cuda)
set(TRT_ROOT /opt/tensorrt)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_ROOT}/include
    ${TRT_ROOT}/include
    src
)

link_directories(
    ${TRT_ROOT}/lib
    ${CUDA_ROOT}/lib64
)

add_executable(yolo_server
    src/main.cpp
    src/detectctrl.cpp
    src/detector.cpp
    src/preprocess.cu
    src/postprocess.cu
)

target_link_libraries(yolo_server
    Drogon::Drogon
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    ${OpenCV_LIBS}
    Threads::Threads
)

target_compile_options(yolo_server PRIVATE -Wno-deprecated-declarations)