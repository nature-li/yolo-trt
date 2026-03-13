#include "detectctrl.hpp"

#include <json/json.h>

#include <chrono>

static const char* COCO_NAMES[] = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

// ─────────────────────────────────────────
// GET /health
// ─────────────────────────────────────────
void DetectCtrl::health(
    const HttpRequestPtr&,
    std::function<void(const HttpResponsePtr&)>&& callback) {
  Json::Value root;
  root["status"] = "ok";
  root["processed"] = (Json::Int64)det_->processed();
  root["ctx_total"] = det_->ctxTotal();
  root["ctx_idle"] = det_->ctxIdle();
  root["ctx_inflight"] = det_->ctxTotal() - det_->ctxIdle();
  callback(HttpResponse::newHttpJsonResponse(root));
}

// ─────────────────────────────────────────
// POST /detect
// multipart/form-data，字段名 "file"
// ─────────────────────────────────────────
void DetectCtrl::detect(
    const HttpRequestPtr& req,
    std::function<void(const HttpResponsePtr&)>&& callback) {
  // 直接读 raw body，客户端发 Content-Type: application/octet-stream
  const auto& body = req->getBody();
  if (body.empty()) {
    Json::Value err;
    err["error"] = "empty body";
    auto resp = HttpResponse::newHttpJsonResponse(err);
    resp->setStatusCode(k400BadRequest);
    callback(resp);
    return;
  }

  std::vector<uint8_t> buf(body.begin(), body.end());
  cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
  if (img.empty()) {
    Json::Value err;
    err["error"] = "cannot decode image";
    auto resp = HttpResponse::newHttpJsonResponse(err);
    resp->setStatusCode(k400BadRequest);
    callback(resp);
    return;
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  /**
   * 扔进推理队列，IO 线程立即返回
   * 推理完成后在推理线程里调 callback，Drogon 负责把响应发回客户端
   */
  bool ok = det_->enqueue(
      img, [callback = std::move(callback), t0](std::vector<Detection> dets) {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        Json::Value root;
        root["latency_ms"] = ms;
        root["detections"] = Json::arrayValue;

        for (const auto& d : dets) {
          const char* name = (d.class_id >= 0 && d.class_id < 80)
                                 ? COCO_NAMES[d.class_id]
                                 : "unknown";
          Json::Value item;
          item["class"] = name;
          item["conf"] = d.conf;
          item["x"] = d.x;
          item["y"] = d.y;
          item["w"] = d.w;
          item["h"] = d.h;
          root["detections"].append(item);
        }

        callback(HttpResponse::newHttpJsonResponse(root));
      });

  if (!ok) {
    Json::Value err;
    err["error"] = "server busy, try again later";
    auto resp = HttpResponse::newHttpJsonResponse(err);
    resp->setStatusCode(k503ServiceUnavailable);
    callback(resp);
    return;
  }
}