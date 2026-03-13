# 编译代码
mkdir -p build
cd build
cmake ..
make -j

# 启动服务
./yolo_server  ../../yolov8n.engine

# 测试接口
curl -X POST http://localhost:8080/detect \
     -H "Content-Type: application/octet-stream" \
     --data-binary @../image.jpg

# 压测接口
wrk -t8 -c100 -d10s -s post.lua http://localhost:8080/detect

# 压测结果
Running 10s test @ http://localhost:8080/detect
  8 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    85.09ms   15.40ms 192.86ms   72.55%
    Req/Sec   140.94     16.62   190.00     68.62%
  11240 requests in 10.00s, 9.62MB read
Requests/sec:   1123.47
Transfer/sec:      0.96MB