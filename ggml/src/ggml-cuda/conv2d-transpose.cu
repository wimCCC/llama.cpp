#include "conv2d-transpose.cuh"
#include "convert.cuh"

template <typename kernel_t>
static __global__ void conv2d_transpose_kernel(const float * __restrict__ input,
                                               const kernel_t * __restrict__ kernel,
                                               float * __restrict__ output,
                                               const int in_w,
                                               const int in_h,
                                               const int out_w,
                                               const int out_h,
                                               const int kernel_w,
                                               const int kernel_h,
                                               const int stride,
                                               const int c_in,
                                               const int c_out,
                                               const int batches) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_elements = out_w * out_h * c_out * batches;

    if (global_idx >= total_elements) {
        return;
    }

    const int out_x_idx = global_idx % out_w;
    const int out_y_idx = (global_idx / out_w) % out_h;
    const int c_idx     = (global_idx / (out_w * out_h)) % c_out;
    const int n_idx     = global_idx / (out_w * out_h * c_out);

    float accumulator = 0;
    // For each output idx, find the inputs that contribute to it by checking stride alignment and bounds

    for (int c_in_idx = 0; c_in_idx < c_in; c_in_idx++) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int in_y = out_y_idx - kh;
            if (in_y < 0 || in_y % stride) {
                continue;
            }
            in_y /= stride;
            if (in_y >= in_h) {
                continue;
            }

            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_x = out_x_idx - kw;
                if (in_x < 0 || in_x % stride) {
                    continue;
                }
                in_x /= stride;
                if (in_x >= in_w) {
                    continue;
                }

                const int input_idx = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + (in_w) *in_y + in_x;
                const int kernel_idx =
                    (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx + (kernel_w) *kh + kw;

                float    input_val = input[input_idx];
                kernel_t kern_val  = kernel[kernel_idx];

                accumulator += input_val * ggml_cuda_cast<float>(kern_val);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + (out_w) *out_y_idx + out_x_idx] = accumulator;
}

static __global__ void convert_f32_to_f16(const float * input, half * output, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i];
    }
}

static __global__ void conv2d_transpose_2x2_pixel_shuffle(
        const float * __restrict__ patches,
        float * __restrict__ output,
        int in_w,
        int in_h,
        int c_out,
        int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }

    const int out_w = 2 * in_w;
    const int out_h = 2 * in_h;
    const int out_x = i % out_w;
    const int out_y = (i / out_w) % out_h;
    const int c = i / ((int64_t) out_w * out_h);
    const int x = out_x / 2;
    const int y = out_y / 2;
    const int q = 2 * (out_y % 2) + out_x % 2;
    output[i] = patches[q + 4 * (c + c_out * (x + in_w * y))];
}

//input is (W, H, C_in, N), Kernel is (W, H, C_out, C_in)
void ggml_cuda_conv_2d_transpose_p0(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const float * input_data  = (const float *) input->data;
    float *       output_data = (float *) dst->data;
    const void *  kernel_data = kernel->data;

    const int input_w      = input->ne[0];
    const int input_h      = input->ne[1];
    const int output_w     = dst->ne[0];
    const int output_h     = dst->ne[1];
    const int channels_in  = input->ne[2];
    const int channels_out = kernel->ne[2];
    const int kernel_w     = kernel->ne[0];
    const int kernel_h     = kernel->ne[1];
    const int stride       = dst->op_params[0];
    const int batches      = input->ne[3];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    cudaStream_t st = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const bool use_2x2_stride_2 = batches == 1 && kernel_w == 2 && kernel_h == 2 && stride == 2 &&
        output_w == 2 * input_w && output_h == 2 * input_h;
    const int total = output_w * output_h * channels_out * batches;
    const int blocks = (total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

    if (use_2x2_stride_2) {
        const int spatial = input_w * input_h;
        const int patch_channels = 4 * channels_out;
        ggml_cuda_pool_alloc<float> patches(ctx.pool(), (size_t) patch_channels * spatial);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t handle = ctx.cublas_handle();
        CUBLAS_CHECK(cublasSetStream(handle, st));

        if (kernel->type == GGML_TYPE_F16) {
            ggml_cuda_pool_alloc<half> input_f16(ctx.pool(), (size_t) spatial * channels_in);
            const int input_blocks = (spatial * channels_in + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) /
                CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;
            convert_f32_to_f16<<<input_blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                input_data, input_f16.get(), (int64_t) spatial * channels_in);
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                patch_channels, spatial, channels_in,
                &alpha, kernel_data, CUDA_R_16F, patch_channels,
                        input_f16.get(), CUDA_R_16F, spatial,
                &beta, patches.get(), CUDA_R_32F, patch_channels,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                patch_channels, spatial, channels_in,
                &alpha, (const float *) kernel_data, patch_channels,
                        input_data, spatial,
                &beta, patches.get(), patch_channels));
        }

        conv2d_transpose_2x2_pixel_shuffle<<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
            patches.get(), output_data, input_w, input_h, channels_out, total);
        return;
    }

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_kernel<half><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
            input_data, (const half *) kernel_data, output_data, input_w, input_h, output_w, output_h, kernel_w,
            kernel_h, stride, channels_in, channels_out, batches);

    } else {
        conv2d_transpose_kernel<float><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
            input_data, (const float *) kernel_data, output_data, input_w, input_h, output_w, output_h, kernel_w,
            kernel_h, stride, channels_in, channels_out, batches);
    }
}
