#pragma once
#include <cuda_runtime.h>

#include <cstdint>

/**
 * 在 GPU 上完成 letterbox resize BGR->RGB  HWC->CHW
 * 参数:
 * - d_src: device 端原图, BGR uint8 HWC 连续内存
 * - d_dst: device 端输出, RGB float32 CHW, 直接指向 d_input_
 */
void gpu_preprocess(const uint8_t* d_src, int src_w, int src_h, float* d_dst,
                    int out_w, int out_h, cudaStream_t stream);