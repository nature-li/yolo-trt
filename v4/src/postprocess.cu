#include "postprocess.cuh"

/**
 * kernel：一个线程处理一个候选框
 * output layout: [84, 8400]
 * row 0~3: cx, cy, w, h
 * row 4~83: 80 个类的置信度
 * num_det: 8400
 * num_cls: 80
 * - input_w: 640
 * - input_h: 640
 */
__global__ void decode_kernel(const float* output, int num_det, int num_cls,
                              float conf_thresh, float pad_x, float pad_y,
                              float scale, int input_w, int input_h,
                              GpuDetection* dets, int* count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_det) {
    return;
  }

  /**
   * output[84][8400]:
   * 第一行代表一个检测框，第一列代表一个检测框的分类置信度
   * cx
   * cy
   * w
   * h
   * class0
   * class79
   */
  float cx = output[0 * num_det + i];
  float cy = output[1 * num_det + i];
  float w = output[2 * num_det + i];
  float h = output[3 * num_det + i];

  /**
   * 找最大类别置信度
   */
  float max_conf = 0.f;
  int max_cls = 0;
  for (int c = 0; c < num_cls; c++) {
    float conf = output[(c + 4) * num_det + i];
    if (conf > max_conf) {
      max_conf = conf;
      max_cls = c;
    }
  }

  if (max_conf < conf_thresh) {
    return;
  }

  /**
   * 把坐标从 640×640 的推理空间还原回原图的归一化坐标
   * active_w: 图像内容的实际宽度
   * active_h: 图像内容的实际高度
   * (cx - pad_x, cy - pad_y): 相当于图像内容的坐标
   */
  // 去掉 letterbox padding, 还原到原图归一化坐标
  float active_w = input_w - 2.f * pad_x;
  float active_h = input_h - 2.f * pad_y;
  float x = (cx - pad_x) / active_w;
  float y = (cy - pad_y) / active_h;
  float nw = w / active_w;
  float nh = h / active_h;

  int slot = atomicAdd(count, 1);
  dets[slot] = {x, y, nw, nh, max_conf, max_cls};
}

/**
 * 对外接口
 * - input_w: 640
 * - input_h: 640
 */
void gpu_decode(const float* d_output, int num_dets, int num_cls,
                float conf_thresh, float pad_x, float pad_y, float scale,
                int input_w, int input_h, GpuDetection* d_dets, int* d_count,
                cudaStream_t stream) {
  // 清零计数器
  cudaMemsetAsync(d_count, 0, sizeof(int), stream);

  int block = 256;
  int grid = (num_dets + block - 1) / block;

  decode_kernel<<<grid, block, 0, stream>>>(d_output, num_dets, num_cls,
                                            conf_thresh, pad_x, pad_y, scale,
                                            input_w, input_h, d_dets, d_count);
}