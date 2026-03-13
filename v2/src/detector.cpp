#include "detector.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

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

  // pinned host buffers
  cudaMallocHost(&h_input_, 1 * 3 * INPUT_H * INPUT_W * sizeof(float));
  cudaMallocHost(&h_output_, 1 * (NUM_CLS + 4) * NUM_DET * sizeof(float));

  context_->setTensorAddress("images", d_input_);
  context_->setTensorAddress("output0", d_output_);
}

Detector::~Detector() {
  cudaStreamSynchronize(stream_);
  cudaStreamDestroy(stream_);

  cudaFree(d_input_);
  cudaFree(d_output_);
  cudaFreeHost(h_input_);
  cudaFreeHost(h_output_);

  delete context_;
  delete engine_;
  delete runtime_;
}

// ─────────────────────────────────────────
// 加载 engine
// ─────────────────────────────────────────
void Detector::loadEngine(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open engine: " + path);

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
// 预处理：letterbox + normalize
// ─────────────────────────────────────────
cv::Mat Detector::preprocess(const cv::Mat& img) {
  // 1. letterbox resize 到 640x640，保持宽高比，填灰边
  int orig_w = img.cols, orig_h = img.rows;
  float scale = std::min((float)INPUT_W / orig_w, (float)INPUT_H / orig_h);
  int new_w = (int)(orig_w * scale);
  int new_h = (int)(orig_h * scale);
  int pad_x = (INPUT_W - new_w) / 2;
  int pad_y = (INPUT_H - new_h) / 2;

  cv::Mat resized, padded;
  cv::resize(img, resized, {new_w, new_h});
  cv::copyMakeBorder(resized, padded, pad_y, INPUT_H - new_h - pad_y, pad_x,
                     INPUT_W - new_w - pad_x, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));

  // 2. BGR → RGB, uint8 → float32, /255.0
  cv::Mat rgb, blob;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(blob, CV_32F, 1.0 / 255.0);

  // 3. HWC → CHW (TensorRT 要求)
  // blob 现在是 [640, 640, 3]，需要变成 [3, 640, 640]
  std::vector<cv::Mat> channels(3);
  cv::split(blob, channels);

  // 拷贝到连续内存：[C, H, W]
  cv::Mat chw(3 * INPUT_H, INPUT_W, CV_32F);
  for (int c = 0; c < 3; c++)
    channels[c].copyTo(chw.rowRange(c * INPUT_H, (c + 1) * INPUT_H));

  return chw;
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

// ─────────────────────────────────────────
// 主接口
// ─────────────────────────────────────────
std::vector<Detection> Detector::detect(const cv::Mat& img, float conf_thresh) {
  constexpr int in_size = 1 * 3 * INPUT_H * INPUT_W;
  constexpr int out_size = 1 * (NUM_CLS + 4) * NUM_DET;

  // 1. 预处理
  cv::Mat chw = preprocess(img);

  // 写入 pinned h_input_
  std::memcpy(h_input_, chw.data, in_size * sizeof(float));

  // 2. H2D 异步：pinned → device，走 DMA，不阻塞 CPU
  cudaMemcpyAsync(d_input_, h_input_, 1 * 3 * INPUT_H * INPUT_W * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // 3. 推理，排进同一条 stream，在 H2D 完成后自动执行
  context_->enqueueV3(stream_);

  // 4.  D2H 异步：推理结束后自动触发，结果落到 pinned h_output_
  cudaMemcpyAsync(h_output_, d_output_, out_size * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);

  // 5. 后处理
  return postprocess(h_output_, img.cols, img.rows, conf_thresh);
}