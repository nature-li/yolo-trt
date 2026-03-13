#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "detector.hpp"

struct BenchResult {
  double mean_ms;
  double median_ms;
  double p99_ms;
  double min_ms;
  double max_ms;
  double fps;
};

BenchResult benchmark(Detector& det, const cv::Mat& img, int warmup = 10,
                      int runs = 200) {
  // warm up，让 GPU 频率稳定
  for (int i = 0; i < warmup; i++) det.detect(img);

  // 用 CUDA event 计时，比 chrono 更准确（不含 CPU 调度抖动）
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<float> latencies(runs);
  for (int i = 0; i < runs; i++) {
    cudaEventRecord(start);
    det.detect(img);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&latencies[i], start, stop);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // 统计
  std::vector<float> sorted = latencies;
  std::sort(sorted.begin(), sorted.end());

  double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
  BenchResult r;
  r.mean_ms = sum / runs;
  r.min_ms = sorted.front();
  r.max_ms = sorted.back();
  r.median_ms = sorted[runs / 2];
  r.p99_ms = sorted[(int)(runs * 0.99)];
  r.fps = 1000.0 / r.mean_ms;

  return r;
}

void printBenchResult(const std::string& tag, const BenchResult& r) {
  printf("\n[BENCH] %s\n", tag.c_str());
  printf("  mean:   %.2f ms\n", r.mean_ms);
  printf("  median: %.2f ms\n", r.median_ms);
  printf("  min:    %.2f ms\n", r.min_ms);
  printf("  max:    %.2f ms\n", r.max_ms);
  printf("  p99:    %.2f ms\n", r.p99_ms);
  printf("  FPS:    %.1f\n", r.fps);
}