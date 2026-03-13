#pragma once

#include <drogon/HttpController.h>

#include "async_detector.hpp"

using namespace drogon;

class DetectCtrl : public HttpController<DetectCtrl, false> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(DetectCtrl::detect, "/detect", Post);
  ADD_METHOD_TO(DetectCtrl::health, "/health", Get);
  METHOD_LIST_END

  explicit DetectCtrl(AsyncDetector* det) : det_(det) {}

  void detect(const HttpRequestPtr& req,
              std::function<void(const HttpResponsePtr&)>&& callback);

  void health(const HttpRequestPtr& req,
              std::function<void(const HttpResponsePtr&)>&& callback);

 private:
  // 全局单例，不拥有
  AsyncDetector* det_;
};
