#include <drogon/drogon.h>

#include "async_detector.hpp"
#include "detectctrl.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <engine> [port]\n", argv[0]);
    return 1;
  }

  std::string engine_path = argv[1];
  int port = (argc >= 3) ? std::stoi(argv[2]) : 8080;

  printf("[server] loading engine: %s\n", engine_path.c_str());

  // AsyncDetector 生命周期和进程一致
  auto detector = std::make_shared<AsyncDetector>(engine_path);

  printf("[server] engine loaded, listening on :%d\n", port);

  drogon::app()
      // IO 线程数，根据并发量调整，推理是单线程所以 IO 线程多了也不会提升吞吐
      .setThreadNum(4)
      .addListener("0.0.0.0", port)
      // 最大上传文件大小 20MB
      .setClientMaxBodySize(20 * 1024 * 1024)
      .setMaxConnectionNum(1000)
      .registerController(std::make_shared<DetectCtrl>(detector.get()))
      .run();

  return 0;
}
