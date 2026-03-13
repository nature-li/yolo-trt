#pragma once
#include <cuda_runtime.h>

#include <cstdint>

struct GpuDetection {
  float x, y, w, h;
  float conf;
  int class_id;
};

/**
 * GPU decode: 过滤 8400 个候选框，写入 d_dets，返回数据到 d_count
 * 参数:
 * - d_output: TRT 输出，device 端 [84, 8400] float32
 * - d_dets: 输出 buffer, device 端，调用方保证足够大（最多 NUM_DET 个)
 * - d_count: 输出标量，device 端，decode 后有效框数量
 */
void gpu_decode(const float* d_output, int num_dets, int num_cls,
                float conf_thresh, float pad_x, float pad_y, float scale,
                int input_w, int input_h, GpuDetection* d_dets, int* d_count,
                cudaStream_t stream);