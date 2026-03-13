#include "detector.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "preprocess.cuh"

// ─────────────────────────────────────────
// 构造 / 析构
// ─────────────────────────────────────────
Detector::Detector(const std::string& engine_path, int pool_size, int max_ctx)
    : max_ctx_(max_ctx) {
  loadEngine(engine_path);

  pool_.reserve(max_ctx);
  for (int i = 0; i < pool_size; i++) {
    auto* ctx = new InferContext();
    initCtx(ctx);
    ctx->detector = this;
    pool_.push_back(ctx);
    total_ctx_++;
  }
}

Detector::~Detector() {
  // 等所有 in-flight ctx 归还
  {
    std::unique_lock<std::mutex> lk(pool_mu_);
    pool_cv_.wait(lk, [&] { return (int)pool_.size() == total_ctx_; });
  }
  for (auto* ctx : pool_) {
    cudaStreamSynchronize(ctx->stream);
    cudaStreamDestroy(ctx->stream);
    cudaFree(ctx->d_input);
    cudaFree(ctx->d_output);
    cudaFree(ctx->d_src);
    cudaFree(ctx->d_dets);
    cudaFree(ctx->d_count);
    cudaFreeHost(ctx->h_dets);
    cudaFreeHost(ctx->h_count);
    delete ctx;
  }

  delete engine_;
  delete runtime_;
}

// ─────────────────────────────────────────
// 加载 engine
// ─────────────────────────────────────────
void Detector::loadEngine(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot open engine: " + path);
  }

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buf(size);
  file.read(buf.data(), size);

  runtime_ = nvinfer1::createInferRuntime(logger_);
  engine_ = runtime_->deserializeCudaEngine(buf.data(), size);

  printf("[Detector] engine loaded: %s\n", path.c_str());
}

/**
 * 初始化单个 InferContext
 */
void Detector::initCtx(InferContext* ctx) {
  // 每个 ctx 独占一个 IExecutionContext，彻底避免并发踩内存
  ctx->context = engine_->createExecutionContext();

  cudaStreamCreate(&ctx->stream);
  cudaMalloc(&ctx->d_input, 1 * 3 * INPUT_H * INPUT_W * sizeof(float));
  cudaMalloc(&ctx->d_output, 1 * (NUM_CLS + 4) * NUM_DET * sizeof(float));
  cudaMalloc(&ctx->d_dets, NUM_DET * sizeof(GpuDetection));
  cudaMalloc(&ctx->d_count, sizeof(int));
  cudaMallocHost(&ctx->h_dets, NUM_DET * sizeof(GpuDetection));
  cudaMallocHost(&ctx->h_count, sizeof(int));

  /**
   * 每个 ctx 独立绑定自己的 buffer
   * 注意：多个 ctx 共用同一个 context_，TRT 支持多 stream 并发
   */
  ctx->context->setTensorAddress("images", ctx->d_input);
  ctx->context->setTensorAddress("output0", ctx->d_output);
}

// ─────────────────────────────────────────
// context pool
// ─────────────────────────────────────────
InferContext* Detector::acquireCtx() {
  std::unique_lock<std::mutex> lk(pool_mu_);
  if (!pool_.empty()) {
    auto* ctx = pool_.back();
    pool_.pop_back();
    return ctx;
  }

  // pool 空了，但还没到上限，动态扩容
  if (total_ctx_ < max_ctx_) {
    auto* ctx = new InferContext();
    initCtx(ctx);
    ctx->detector = this;
    total_ctx_++;
    return ctx;  // 直接返回，不进 pool
  }

  // 到上限了，等归还
  pool_cv_.wait(lk, [&] { return !pool_.empty(); });

  auto* ctx = pool_.back();
  pool_.pop_back();
  return ctx;
}

void Detector::releaseCtx(InferContext* ctx) {
  {
    std::lock_guard<std::mutex> lk(pool_mu_);
    pool_.push_back(ctx);
  }
  pool_cv_.notify_one();
}

/**
 * GPU 完成回调
 * 在 CUDA 内部回调线程触发，不是推理线程
 */
void CUDART_CB Detector::onGpuDone(cudaStream_t, cudaError_t status,
                                   void* userdata) {
  auto* ctx = reinterpret_cast<InferContext*>(userdata);
  auto* det = reinterpret_cast<Detector*>(ctx->detector);

  std::vector<Detection> results;
  if (status == cudaSuccess) {
    results = det->cpu_nms(ctx->h_dets, *ctx->h_count);
  }

  /**
   * 先还 ctx，再触发 callback
   * 避免 callback 耗时长时导致 pool 耗尽
   */
  auto cb = std::move(ctx->callback);
  det->releaseCtx(ctx);
  cb(std::move(results));

  det->processed_++; 
}

/**
 * 异步推理主接口
 */
void Detector::detectAsync(
    const cv::Mat& img, std::function<void(std::vector<Detection>)> callback) {
  // 从 pool 取 ctx，pool 空时阻塞等待
  InferContext* ctx = acquireCtx();
  ctx->callback = std::move(callback);

  // 按需扩容 d_src
  size_t src_bytes = img.cols * img.rows * 3;
  if (src_bytes > ctx->d_src_size) {
    cudaFree(ctx->d_src);
    cudaMalloc(&ctx->d_src, src_bytes);
    ctx->d_src_size = src_bytes;
  }

  // letterbox 参数
  float scale = std::min((float)INPUT_W / img.cols, (float)INPUT_H / img.rows);
  int new_w = std::min((int)(img.cols * scale), INPUT_W);
  int new_h = std::min((int)(img.rows * scale), INPUT_H);
  float pad_x = (INPUT_W - new_w) / 2.f;
  float pad_y = (INPUT_H - new_h) / 2.f;

  /**
   * 全部入队到 ctx->stream，不等待
   */
  // 1. H2D 原图
  cudaMemcpyAsync(ctx->d_src, img.data, src_bytes, cudaMemcpyHostToDevice,
                  ctx->stream);

  // 2. GPU preprocess
  gpu_preprocess(ctx->d_src, img.cols, img.rows, (float*)ctx->d_input, INPUT_W,
                 INPUT_H, ctx->stream);

  // 3. 推理
  ctx->context->enqueueV3(ctx->stream);

  // 4. GPU decode
  gpu_decode((float*)ctx->d_output, NUM_DET, NUM_CLS, 0.25f, pad_x, pad_y,
             scale, INPUT_W, INPUT_H, ctx->d_dets, ctx->d_count, ctx->stream);

  // 5. D2H decode 结果
  cudaMemcpyAsync(ctx->h_count, ctx->d_count, sizeof(int),
                  cudaMemcpyDeviceToHost, ctx->stream);
  cudaMemcpyAsync(ctx->h_dets, ctx->d_dets, NUM_DET * sizeof(GpuDetection),
                  cudaMemcpyDeviceToHost, ctx->stream);

  // 6. 注册完成回调，提交后立即返回
  cudaStreamAddCallback(ctx->stream, onGpuDone, ctx, 0);
}

// ─────────────────────────────────────────
// CPU NMS
// ─────────────────────────────────────────
std::vector<Detection> Detector::cpu_nms(GpuDetection* dets, int count,
                                         float iou_thresh) {
  if (count <= 0) return {};
  std::sort(dets, dets + count,
            [](const GpuDetection& a, const GpuDetection& b) {
              return a.conf > b.conf;
            });

  auto iou_fn = [](const GpuDetection& a, const GpuDetection& b) {
    float ax1 = a.x - a.w / 2, ay1 = a.y - a.h / 2;
    float ax2 = a.x + a.w / 2, ay2 = a.y + a.h / 2;
    float bx1 = b.x - b.w / 2, by1 = b.y - b.h / 2;
    float bx2 = b.x + b.w / 2, by2 = b.y + b.h / 2;
    float inter = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1)) *
                  std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-6f);
  };

  std::vector<Detection> out;
  std::vector<bool> suppressed(count, false);
  for (int i = 0; i < count; i++) {
    if (suppressed[i]) {
      continue;
    }

    auto& d = dets[i];
    out.push_back({d.x, d.y, d.w, d.h, d.conf, d.class_id});
    for (int j = i + 1; j < count; j++) {
      if (!suppressed[j] && dets[i].class_id == dets[j].class_id) {
        if (iou_fn(dets[i], dets[j]) > iou_thresh) {
          suppressed[j] = true;
        }
      }
    }
  }

  return out;
}