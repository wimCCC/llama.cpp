#include "pool2d.cuh"

template <typename Ti, typename To>
static  __global__ void pool2d_nchw_kernel(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int I_HW = ih * iw;
    const int O_HW = oh * ow;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / ow;
    const int cur_ow = idx % O_HW % ow;
    const Ti* i_ptr = src + nc * I_HW;
    To* o_ptr = dst + nc * O_HW;
    const int start_h = cur_oh * sh - ph;
    const int bh = max(0, start_h);
    const int eh = min(ih, start_h + kh);
    const int start_w = cur_ow * sw - pw;
    const int bw = max(0, start_w);
    const int ew = min(iw, start_w + kw);
    const To scale = 1. / (kh * kw);
    To res = 0;

    switch (op) {
        case GGML_OP_POOL_AVG: res = 0; break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: assert(false);
    }

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
#if __CUDA_ARCH__ >= 350
            Ti cur = __ldg(i_ptr + i * iw + j);
#else
            Ti cur = i_ptr[i * iw + j];
#endif
            switch (op) {
                case GGML_OP_POOL_AVG: res += cur * scale; break;
                case GGML_OP_POOL_MAX: res = max(res, (To)cur); break;
                default: assert(false);
            }
        }
    }
    o_ptr[cur_oh * ow + cur_ow] = res;
}

static void pool2d_nchw_kernel_f32_f32_cuda(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {

    const int num_blocks = (parallel_elements + CUDA_POOL2D_BLOCK_SIZE - 1) / CUDA_POOL2D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool2d_nchw_kernel<<<block_nums, CUDA_POOL2D_BLOCK_SIZE, 0, stream>>>(ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    const int64_t IH = src0->ne[1];
    const int64_t IW = src0->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const int parallel_elements = N * OC * OH * OW;

    pool2d_nchw_kernel_f32_f32_cuda(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0, parallel_elements, src0_d, dst_d, op, stream);
}

static __global__ void roi_align_f32(
        const float * src, const float * boxes, float * dst,
    int64_t channels, int64_t width, int64_t height, int64_t pooled_width, int64_t pooled_height,
        float spatial_scale_x, float spatial_scale_y, int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }

    const int64_t c = i % channels;
    const int64_t bin = (i / channels) % (pooled_width * pooled_height);
    const int64_t roi = i / (channels * pooled_width * pooled_height);
    const int64_t bx = bin % pooled_width;
    const int64_t by = bin / pooled_width;
    const float * box = boxes + 4 * roi;
    const float x0 = box[0] * spatial_scale_x;
    const float y0 = box[1] * spatial_scale_y;
    const float x1 = box[2] * spatial_scale_x;
    const float y1 = box[3] * spatial_scale_y;
    const float x = max(0.0f, min((float) (width - 1),
        x0 + (bx + 0.5f) * (x1 - x0) / pooled_width - 0.5f));
    const float y = max(0.0f, min((float) (height - 1),
        y0 + (by + 0.5f) * (y1 - y0) / pooled_height - 0.5f));
    const int64_t xa = (int64_t) floorf(x);
    const int64_t ya = (int64_t) floorf(y);
    const int64_t xb = min(xa + 1, width - 1);
    const int64_t yb = min(ya + 1, height - 1);
    const float wx = x - xa;
    const float wy = y - ya;
    const float v00 = src[c + channels * (xa + width * ya)];
    const float v10 = src[c + channels * (xb + width * ya)];
    const float v01 = src[c + channels * (xa + width * yb)];
    const float v11 = src[c + channels * (xb + width * yb)];

    dst[i] = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v10)
           + wy * ((1.0f - wx) * v01 + wx * v11);
}

void ggml_cuda_op_roi_align(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src   = dst->src[0];
    const ggml_tensor * boxes = dst->src[1];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(boxes->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(boxes));
    GGML_ASSERT(src->ne[3] == 1 && boxes->ne[0] == 4);

    const int64_t total = ggml_nelements(dst);
    const int num_blocks = (total + CUDA_POOL2D_BLOCK_SIZE - 1) / CUDA_POOL2D_BLOCK_SIZE;
    roi_align_f32<<<num_blocks, CUDA_POOL2D_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) src->data, (const float *) boxes->data, (float *) dst->data,
        src->ne[0], src->ne[1], src->ne[2], dst->ne[1], dst->ne[2],
        ggml_get_op_params_f32(dst, 0), ggml_get_op_params_f32(dst, 1), total);
}
