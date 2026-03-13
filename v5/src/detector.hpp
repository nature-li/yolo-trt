#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
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

/**
 * 每个 in-flight 推理请求独占的资源
 */
struct InferContext {
  // 每个 ctx 独占一个 IExecutionContext，避免多线程并发 enqueueV3 踩内存
  nvinfer1::IExecutionContext* context = nullptr;

  void* d_input = nullptr;
  void* d_output = nullptr;
  uint8_t* d_src = nullptr;
  size_t d_src_size = 0;
  GpuDetection* d_dets = nullptr;
  int* d_count = nullptr;
  GpuDetection* h_dets = nullptr;
  int* h_count = nullptr;

  cudaStream_t stream = nullptr;

  // 推理完成后的回调
  std::function<void(std::vector<Detection>)> callback;

  // 反向指针，回调里做 NMS 用
  void* detector = nullptr;
};

class Detector {
 public:
  explicit Detector(const std::string& engine_path, int pool_size = 4,
                    int max_ctx = 64);
  ~Detector();

  /**
   * 异步推理:
   * 提交后立即返回，GPU 完成后回调 callback
   * callback 在 CUDA 回调线程里触发 (非 IO 线程, 非推理线程)
   */
  void detectAsync(const cv::Mat& img,
                   std::function<void(std::vector<Detection>)> callback);
  /**
   * 从 pool 取/还 context (供 async_detector 使用)
   */
  InferContext* acquireCtx();
  void releaseCtx(InferContext* ctx);

  /**
   * NMS 保持在 CPU，供回调调用
   */
  std::vector<Detection> cpu_nms(GpuDetection* dets, int count,
                                 float iou_thresh = 0.45f);

  int ctxTotal() const { return total_ctx_.load(); }
  int ctxIdle() const {
    std::lock_guard<std::mutex> lk(pool_mu_);
    return (int)pool_.size();
  }
  long processed() const { return processed_.load(); }

 private:
  void loadEngine(const std::string& path);
  void initCtx(InferContext* ctx);
  static void CUDART_CB onGpuDone(cudaStream_t stream, cudaError_t error,
                                  void* userdata);

 private:
  Logger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;

  // context pool
  std::vector<InferContext*> pool_;
  mutable std::mutex pool_mu_;
  std::condition_variable pool_cv_;

  int max_ctx_ = 0;  // 上限，防止显存爆
  std::atomic<int> total_ctx_{0};
  std::atomic<long> processed_{0};

  /**
   * yolov8n: input[1,3,640,640], output[1,84,8400]
   */
  static constexpr int INPUT_H = 640;
  static constexpr int INPUT_W = 640;
  static constexpr int NUM_DET = 8400;
  static constexpr int NUM_CLS = 80;
};