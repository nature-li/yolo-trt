#include "detector.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "preprocess.cuh"

// ─────────────────────────────────────────
// 构造 / 析构
// ─────────────────────────────────────────
Detector::Detector(const std::string& engine_path) {
  loadEngine(engine_path);

  cudaStreamCreate(&stream_);

  // 分配 device buffer
  // input:  1 * 3 * 640 * 640 * sizeof(float)
  // output: 1 * 84 * 8400 * sizeof(float)
  cudaMalloc(&d_input_, 1 * 3 * INPUT_H * INPUT_W * sizeof(float));
  cudaMalloc(&d_output_, 1 * (NUM_CLS + 4) * NUM_DET * sizeof(float));

  // d_src_ 按需分配，构造时先不分配
  d_src_ = nullptr;
  d_src_size_ = 0;

  // decode output buffers
  cudaMalloc(&d_dets_, NUM_DET * sizeof(GpuDetection));
  cudaMalloc(&d_count_, sizeof(int));
  cudaMallocHost(&h_dets_, NUM_DET * sizeof(GpuDetection));
  cudaMallocHost(&h_count_, sizeof(int));

  context_->setTensorAddress("images", d_input_);
  context_->setTensorAddress("output0", d_output_);
}

Detector::~Detector() {
  cudaStreamSynchronize(stream_);
  cudaStreamDestroy(stream_);

  cudaFree(d_input_);
  cudaFree(d_output_);
  cudaFree(d_src_);
  cudaFree(d_dets_);
  cudaFree(d_count_);
  cudaFreeHost(h_dets_);
  cudaFreeHost(h_count_);

  delete context_;
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
  context_ = engine_->createExecutionContext();

  printf("[Detector] engine loaded: %s\n", path.c_str());
}

// ─────────────────────────────────────────
// NMS
// ─────────────────────────────────────────
static float iou(const Detection& a, const Detection& b) {
  float ax1 = a.x - a.w / 2, ay1 = a.y - a.h / 2;
  float ax2 = a.x + a.w / 2, ay2 = a.y + a.h / 2;
  float bx1 = b.x - b.w / 2, by1 = b.y - b.h / 2;
  float bx2 = b.x + b.w / 2, by2 = b.y + b.h / 2;

  float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
  float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
  float a_area = a.w * a.h, b_area = b.w * b.h;
  return inter / (a_area + b_area - inter + 1e-6f);
}

static std::vector<Detection> nms(std::vector<Detection> dets,
                                  float iou_thresh = 0.45f) {
  std::sort(
      dets.begin(), dets.end(),
      [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
  std::vector<Detection> out;
  std::vector<bool> suppressed(dets.size(), false);
  for (size_t i = 0; i < dets.size(); i++) {
    if (suppressed[i]) continue;
    out.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); j++)
      if (!suppressed[j] && dets[i].class_id == dets[j].class_id)
        if (iou(dets[i], dets[j]) > iou_thresh) suppressed[j] = true;
  }
  return out;
}

// ─────────────────────────────────────────
// 后处理：解析 [1, 84, 8400]
// ─────────────────────────────────────────
// YOLOv8 输出格式：
//   output[0][0~3][i]  = cx, cy, w, h  (归一化到640x640)
//   output[0][4~83][i] = 80个类别的置信度（无objectness）
std::vector<Detection> Detector::postprocess(float* output, int orig_w,
                                             int orig_h, float conf_thresh) {
  float scale = std::min((float)INPUT_W / orig_w, (float)INPUT_H / orig_h);
  int pad_x = (INPUT_W - (int)(orig_w * scale)) / 2;
  int pad_y = (INPUT_H - (int)(orig_h * scale)) / 2;

  std::vector<Detection> dets;

  for (int i = 0; i < NUM_DET; i++) {
    // output layout: [84, 8400]，列优先访问
    float cx = output[0 * NUM_DET + i];
    float cy = output[1 * NUM_DET + i];
    float w = output[2 * NUM_DET + i];
    float h = output[3 * NUM_DET + i];

    // 找最大类别置信度
    float max_conf = 0.f;
    int max_cls = 0;
    for (int c = 0; c < NUM_CLS; c++) {
      float conf = output[(4 + c) * NUM_DET + i];
      if (conf > max_conf) {
        max_conf = conf;
        max_cls = c;
      }
    }

    if (max_conf < conf_thresh) continue;

    // 去掉 letterbox padding，还原到原图坐标（归一化）
    float x = (cx - pad_x) / (INPUT_W - 2 * pad_x);
    float y = (cy - pad_y) / (INPUT_H - 2 * pad_y);
    float nw = w / (INPUT_W - 2 * pad_x);
    float nh = h / (INPUT_H - 2 * pad_y);

    dets.push_back({x, y, nw, nh, max_conf, max_cls});
  }

  return nms(dets);
}

std::vector<Detection> Detector::cpu_nms(GpuDetection* dets, int count,
                                         float iou_thresh) {
  // 按置信度降序排列
  std::sort(dets, dets + count,
            [](const GpuDetection& a, const GpuDetection& b) {
              return a.conf > b.conf;
            });

  auto iou_fn = [](const GpuDetection& a, const GpuDetection& b) {
    float ax1 = a.x - a.w / 2;
    float ay1 = a.y - a.h / 2;
    float ax2 = a.x + a.w / 2;
    float ay2 = a.y + a.h / 2;
    float bx1 = b.x - b.w / 2;
    float by1 = b.y - b.h / 2;
    float bx2 = b.x + b.w / 2;
    float by2 = b.y + b.h / 2;

    float ix1 = std::max(ax1, bx1);
    float iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2);
    float iy2 = std::min(ay2, by2);

    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
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

// ─────────────────────────────────────────
// 主接口
// ─────────────────────────────────────────
std::vector<Detection> Detector::detect(const cv::Mat& img, float conf_thresh) {
  constexpr int out_size = 1 * (NUM_CLS + 4) * NUM_DET;

  /**
   * 1. 预处理
   * 按需分配/重用 d_src_ (原图大小不固定)
   */
  size_t src_bytes = img.cols * img.rows * 3;
  if (src_bytes > d_src_size_) {
    cudaFree(d_src_);
    cudaMalloc(&d_src_, src_bytes);
    d_src_size_ = src_bytes;
  }

  /**
   * 计算 letterbox 参数 (GPU decode 需要)
   */
  float scale = std::min((float)INPUT_W / img.cols, (float)INPUT_H / img.rows);
  int new_w = std::min((int)(img.cols * scale), INPUT_W);
  int new_h = std::min((int)(img.rows * scale), INPUT_H);
  float pad_x = (INPUT_W - new_w) / 2.f;
  float pad_y = (INPUT_H - new_h) / 2.f;

  /**
   * 2.原图 H2D (BGR uint8, 连续内存)
   * 这是唯一一次从 host 到 device 的拷贝
   */
  cudaMemcpyAsync(d_src_, img.data, src_bytes, cudaMemcpyHostToDevice, stream_);

  /**
   * 3. GPU preprocess
   * letterbox BRG->RGB HWC->CHW
   * 直接写到 d_input_, stream 上排在 H2D 后
   */
  gpu_preprocess(d_src_, img.cols, img.rows, (float*)d_input_, INPUT_W, INPUT_H,
                 stream_);

  // 3. 推理，排进同一条 stream，在 H2D 完成后自动执行
  context_->enqueueV3(stream_);

  /**
   * GPU decode:
   * 过滤候选框，写入 d_dets_ / d_count_
   */
  gpu_decode((float*)d_output_, NUM_DET, NUM_CLS, conf_thresh, pad_x, pad_y,
             scale, INPUT_W, INPUT_H, d_dets_, d_count_, stream_);

  /**
   * D2H: 只拷贝 decode 结果，比拷贝整个 output 小很多
   */
  cudaMemcpyAsync(h_count_, d_count_, sizeof(int), cudaMemcpyDeviceToHost,
                  stream_);
  cudaMemcpyAsync(h_dets_, d_dets_, NUM_DET * sizeof(GpuDetection),
                  cudaMemcpyDeviceToHost, stream_);

  /**
   * 等 strean 完成
   */
  cudaStreamSynchronize(stream_);

  /**
   * 后处理
   */
  return cpu_nms(h_dets_, *h_count_, 0.45f);
}