cmake_minimum_required(VERSION 3.18)
project(yolo_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

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

add_executable(yolo_infer 
    src/main.cpp 
    src/detector.cpp 
    src/preprocess.cu 
    src/postprocess.cu
)

target_link_libraries(yolo_infer
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    ${OpenCV_LIBS}
)

target_compile_options(yolo_infer PRIVATE -Wno-deprecated-declarations)