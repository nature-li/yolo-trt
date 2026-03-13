#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "detector.hpp"

/**
 * 推理请求
 */
struct InferRequest {
  cv::Mat img;
  std::function<void(std::vector<Detection>)> callback;
};

/**
 * AsyncDetector
 * 推理线程:
 * 1.取队列
 * 2.detectAsync()
 * 3.立即取下一张
 * 4.GPU 完成后 CUDA 回调线程触发 callback
 */
class AsyncDetector {
 public:
  explicit AsyncDetector(const std::string& engine_path, int pool_size = 32,
                         int num_workers = 4, int max_queue = 100)
      : det_(engine_path, pool_size), max_queue_(max_queue), running_(true) {
    for (int i = 0; i < num_workers; i++) {
      workers_.emplace_back(&AsyncDetector::workerLoop, this);
    }
  }

  ~AsyncDetector() {
    running_ = false;
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  bool enqueue(const cv::Mat& img,
               std::function<void(std::vector<Detection>)> callback) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if ((int)queue_.size() >= max_queue_) {
        return false;
      }
      queue_.push({img.clone(), std::move(callback)});
    }
    cv_.notify_one();
    return true;
  }

  int ctxTotal() const { return det_.ctxTotal(); }
  int ctxIdle() const { return det_.ctxIdle(); }
  long processed() const { return det_.processed(); }

 private:
  void workerLoop() {
    while (running_) {
      InferRequest req;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !queue_.empty() || !running_; });
        if (!running_ && queue_.empty()) {
          break;
        }

        req = std::move(queue_.front());
        queue_.pop();
      }

      /**
       * detectAsync 提交 GPU 任务后立即返回
       * 推理线程马上去取下一张，不等 GPU
       */
      det_.detectAsync(req.img, std::move(req.callback));
    }
  }

 private:
  Detector det_;
  std::vector<std::thread> workers_;
  std::atomic<bool> running_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<InferRequest> queue_;
  int max_queue_;
};
