#include "preprocess.cuh"

/**
 * 每个线程处理一个像素
 * 参数:
 * -src: BGR uint8 HWC，原始尺寸
 */
__global__ void letterbox_kernel(const uint8_t* src, int src_w, int src_h,
                                 float* dst, int out_w, int out_h, int pad_x,
                                 int pad_y, int new_w, int new_h, float scale) {
  /**
   * output col, row
   */
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= out_w || dy >= out_h) {
    return;
  }

  float r;
  float g;
  float b;

  // padding 区域：填灰
  if (dx < pad_x || dx >= pad_x + new_w || dy < pad_y || dy >= pad_y + new_h) {
    r = 114.f / 255.f;
    g = 114.f / 255.f;
    b = 114.f / 255.f;
  } else {
    // 还原到原图坐标（最近邻）
    int sx = (int)((dx - pad_x) / scale);
    int sy = (int)((dy - pad_y) / scale);
    sx = max(0, min(sx, src_w - 1));
    sy = max(0, min(sy, src_h - 1));

    // BGR -> RGB
    const uint8_t* px = src + (sy * src_w + sx) * 3;
    b = px[0] / 255.f;
    g = px[1] / 255.f;
    r = px[2] / 255.f;
  }

  /**
   * 写 CHW: plane 0=R, 1=G, 2=B
   */
  int plane = out_w * out_h;
  dst[0 * plane + dy * out_w + dx] = r;
  dst[1 * plane + dy * out_w + dx] = g;
  dst[2 * plane + dy * out_w + dx] = b;
}

/**
 * 对外接口
 */
void gpu_preprocess(const uint8_t* d_src,  // device, BGR HWC uint8
                    int src_w, int src_h,
                    float* d_dst,  // device, RGB CHW float32, 即 d_input_
                    int out_w, int out_h, cudaStream_t stream) {
  float scale = fminf((float)out_w / src_w, (float)out_h / src_h);
  int new_w = (int)(src_w * scale);
  int new_h = (int)(src_h * scale);
  int pad_x = (out_w - new_w) / 2;
  int pad_y = (out_h - new_h) / 2;

  /**
   * 16x16 block
   */
  dim3 block(16, 16);
  dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);

  letterbox_kernel<<<grid, block, 0, stream>>>(d_src, src_w, src_h, d_dst,
                                               out_w, out_h, pad_x, pad_y,
                                               new_w, new_h, scale);
}