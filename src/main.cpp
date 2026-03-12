#include <chrono>
#include <iostream>

#include "benchmark.hpp"
#include "detector.hpp"

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

int main1(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Usage: %s <engine> <image>\n", argv[0]);
    return 1;
  }

  Detector det(argv[1]);

  cv::Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    printf("cannot open image\n");
    return 1;
  }

  // warm up
  for (int i = 0; i < 3; i++) {
    det.detect(img);
  }

  // benchmark
  auto t0 = std::chrono::high_resolution_clock::now();
  int N = 100;
  std::vector<Detection> results;
  for (int i = 0; i < N; i++) {
    results = det.detect(img);
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
  printf("latency: %.2f ms (%.0f FPS)\n", ms, 1000.0 / ms);
  printf("detections: %zu\n", results.size());

  // 画框保存
  for (auto& d : results) {
    int x1 = (d.x - d.w / 2) * img.cols;
    int y1 = (d.y - d.h / 2) * img.rows;
    int x2 = (d.x + d.w / 2) * img.cols;
    int y2 = (d.y + d.h / 2) * img.rows;

    const char* name = COCO_NAMES[d.class_id];

    cv::rectangle(img, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);
    cv::putText(img, name, {x1, y1 - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                {0, 255, 0}, 1);
    printf("  class_id=%d conf=%.2f box=[%d,%d,%d,%d] class_name=%s\n",
           d.class_id, d.conf, x1, y1, x2, y2, name);
  }
  cv::imwrite("result.jpg", img);
  printf("saved result.jpg");

  return 0;
}

int main(int argc, char** argv) {
  Detector det(argv[1]);
  cv::Mat img = cv::imread(argv[2]);

  // baseline benchmark
  auto r = benchmark(det, img);
  printBenchResult("baseline (CPU preprocess)", r);

  // 顺便跑一次看检测结果
  auto results = det.detect(img);
  for (auto& d : results)
    printf("  [%s] conf=%.2f\n", COCO_NAMES[d.class_id], d.conf);

  cv::imwrite("result.jpg", img);
  return 0;
}