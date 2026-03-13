#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "postprocess.cuh"
#include "preprocess.cuh"

// 检测结果
struct Detection {
  float x, y, w, h;  // 归一化坐标 center_x, center_y, width, height
  float conf;
  int class_id;
};

// TRT logger
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    // 不打印 info 和 verbose
    if (severity <= Severity::kWARNING) {
      printf("[TRT]: %s\n", msg);
    }
  }
};

class Detector {
 public:
  explicit Detector(const std::string& engine_path);
  ~Detector();

  // 主接口: 输入 BGR 图像，返回检测结果
  std::vector<Detection> detect(const cv::Mat& img, float conf_thresh = 0.25f);

 private:
  void loadEngine(const std::string& path);
  std::vector<Detection> cpu_nms(GpuDetection* dets, int count,
                                 float iou_thresh);
  std::vector<Detection> postprocess(float* output, int orig_w, int orig_h,
                                     float conf_thresh);

 private:
  Logger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;

  cudaStream_t stream_ = nullptr;

  void* d_input_ = nullptr;
  void* d_output_ = nullptr;

  // d_src_: 原图上传到 GPU 的暂存 buffer (BGR uint8)
  uint8_t* d_src_ = nullptr;
  // 当前分配的字节数，按需 realloc
  size_t d_src_size_ = 0;

  // decode 输出 buffer (device + pinned host)
  GpuDetection* d_dets_ = nullptr;
  int* d_count_ = nullptr;          // device
  GpuDetection* h_dets_ = nullptr;  // pinned
  int* h_count_ = nullptr;          // pinned

  /**
   * yolov8n: input[1,3,640,640], output[1,84,8400]
   */
  static constexpr int INPUT_H = 640;
  static constexpr int INPUT_W = 640;
  static constexpr int NUM_DET = 8400;
  static constexpr int NUM_CLS = 80;
};