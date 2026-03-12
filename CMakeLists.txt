cmake_minimum_required(VERSION 3.18)
project(yolo_trt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)

set(TRT_ROOT /opt/tensorrt)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TRT_ROOT}/include
    src
)

link_directories(${TRT_ROOT}/lib)

add_executable(yolo_infer src/main.cpp)

target_link_libraries(yolo_infer
    nvinfer
    nvinfer_plugin
    nvonnxparser
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)