#include "models.h"

namespace {
struct roi_align_params {
    int width;
    int height;
    int pooled;
    int n_rois;
    float spatial_scale_x;
    float spatial_scale_y;
};

static void roi_align_compute(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * p = (const roi_align_params *) userdata;
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * boxes = dst->src[1];
    const float * x = (const float *) ggml_get_data(src);
    const float * r = (const float *) ggml_get_data(boxes);
    float * out = (float *) ggml_get_data(dst);
    const int channels = src->ne[0];
    const int bins = p->pooled * p->pooled;
    const int total = channels * bins * p->n_rois;
    for (int i = ith; i < total; i += nth) {
        const int c = i % channels;
        const int q = i / channels;
        const int roi = q / bins;
        const int bin = q % bins;
        const int by = bin / p->pooled;
        const int bx = bin % p->pooled;
        const float * b = r + 4 * roi;
        const float x0 = b[0] * p->spatial_scale_x;
        const float y0 = b[1] * p->spatial_scale_y;
        const float x1 = b[2] * p->spatial_scale_x;
        const float y1 = b[3] * p->spatial_scale_y;
        const float xx = x0 + (bx + 0.5f) * (x1 - x0) / p->pooled - 0.5f;
        const float yy = y0 + (by + 0.5f) * (y1 - y0) / p->pooled - 0.5f;
        const float fx = std::max(0.0f, std::min((float) (p->width - 1), xx));
        const float fy = std::max(0.0f, std::min((float) (p->height - 1), yy));
        const int xa = (int) floorf(fx), ya = (int) floorf(fy);
        const int xb = std::min(xa + 1, p->width - 1), yb = std::min(ya + 1, p->height - 1);
        const float wx = fx - xa, wy = fy - ya;
        auto at = [&](int px, int py) { return x[c + channels * (px + p->width * py)]; };
        out[i] = (1 - wy) * ((1 - wx) * at(xa, ya) + wx * at(xb, ya))
               + wy * ((1 - wx) * at(xa, yb) + wx * at(xb, yb));
    }
}

static ggml_tensor * roi_align(ggml_context * ctx, ggml_tensor * x, ggml_tensor * boxes,
        int width, int height, int pooled, int n_rois, float spatial_scale_x, float spatial_scale_y) {
    auto * p = new roi_align_params{ width, height, pooled, n_rois, spatial_scale_x, spatial_scale_y };
    ggml_tensor * args[] = { x, boxes };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], pooled, pooled, n_rois,
        args, 2, roi_align_compute, GGML_N_TASKS_MAX, p);
}
static ggml_tensor * davit_dwconv(ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * w, ggml_tensor * b, int width, int height, int channels) {
    ggml_tensor * image = ggml_reshape_4d(ctx, x, channels, width, height, 1);
    image = ggml_cont(ctx, ggml_permute(ctx, image, 2, 0, 1, 3));
    image = ggml_conv_2d_dw(ctx, w, image, 1, 1, 1, 1, 1, 1);
    if (b) image = ggml_add(ctx, image, ggml_reshape_4d(ctx, b, 1, 1, channels, 1));
    image = ggml_cont(ctx, ggml_permute(ctx, image, 1, 2, 0, 3));
    return ggml_cont_2d(ctx, image, channels, width * height);
}

static ggml_tensor * davit_channel_attn(clip_graph * graph, const clip_aux_davit_block & layer,
        ggml_tensor * x, int channels, int groups, int n_tokens) {
    ggml_context * ctx = graph->ctx0;
    ggml_tensor * qkv = graph->build_mm(layer.channel_qkv_w, x);
    qkv = ggml_add(ctx, qkv, layer.channel_qkv_b);
    const int dim = channels / groups;
    const size_t row = ggml_row_size(qkv->type, channels);
    ggml_tensor * q = ggml_view_2d(ctx, qkv, channels, n_tokens, qkv->nb[1], 0);
    ggml_tensor * k = ggml_view_2d(ctx, qkv, channels, n_tokens, qkv->nb[1], row);
    ggml_tensor * v = ggml_view_2d(ctx, qkv, channels, n_tokens, qkv->nb[1], row * 2);
    auto qk_view = [&](ggml_tensor * t) {
        t = ggml_reshape_3d(ctx, ggml_cont(ctx, t), dim, groups, n_tokens);
        return ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3));
    };
    q = qk_view(q); k = qk_view(k);
    v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), dim, groups, n_tokens);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / sqrtf((float) n_tokens), 0.0f);
    ggml_tensor * out = ggml_mul_mat(ctx, v, scores);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    out = ggml_cont_2d(ctx, out, channels, n_tokens);
    out = graph->build_mm(layer.channel_o_w, out);
    return ggml_add(ctx, out, layer.channel_o_b);
}

static ggml_tensor * davit_window_tokens(ggml_context * ctx, ggml_tensor * x,
        int width, int height, int channels, int window, int & nwin_x, int & nwin_y) {
    nwin_x = (width + window - 1) / window;
    nwin_y = (height + window - 1) / window;
    x = ggml_cont_3d(ctx, x, channels, width, height);
    std::vector<ggml_tensor *> windows;
    for (int wy = 0; wy < nwin_y; ++wy) for (int wx = 0; wx < nwin_x; ++wx) {
        const int w = std::min(window, width - wx * window);
        const int h = std::min(window, height - wy * window);
        ggml_tensor * part = ggml_view_3d(ctx, x, channels, w, h, x->nb[1], x->nb[2],
            (size_t) wx * window * x->nb[1] + (size_t) wy * window * x->nb[2]);
        if (w != window || h != window) part = ggml_pad(ctx, part, 0, window - w, window - h, 0);
        windows.push_back(ggml_cont_2d(ctx, part, channels, window * window));
    }
    ggml_tensor * result = windows[0];
    for (size_t i = 1; i < windows.size(); ++i) result = ggml_concat(ctx, result, windows[i], 1);
    return result;
}

static ggml_tensor * davit_window_reverse(ggml_context * ctx, ggml_tensor * x,
        int width, int height, int channels, int window, int nwin_x, int nwin_y) {
    std::vector<ggml_tensor *> rows;
    for (int wy = 0; wy < nwin_y; ++wy) {
        ggml_tensor * row = nullptr;
        for (int wx = 0; wx < nwin_x; ++wx) {
            ggml_tensor * part = ggml_view_2d(ctx, x, channels, window * window,
                x->nb[1], ((size_t) wy * nwin_x + wx) * window * window * x->nb[1]);
            part = ggml_cont_3d(ctx, part, channels, window, window);
            row = row ? ggml_concat(ctx, row, part, 1) : part;
        }
        rows.push_back(row);
    }
    ggml_tensor * image = rows[0];
    for (size_t i = 1; i < rows.size(); ++i) image = ggml_concat(ctx, image, rows[i], 2);
    image = ggml_cont(ctx, image);
    image = ggml_view_3d(ctx, image, channels, width, height, image->nb[1], image->nb[2], 0);
    return ggml_cont_2d(ctx, image, channels, width * height);
}

}

ggml_cgraph * clip_graph_vlm_fo1_aux::build() {
    ggml_tensor * inp_raw = build_inp_raw();
    const int n_tokens = hparams.n_region_tokens;
    const bool has_main_features = encode_params && encode_params->input_feature_map;
    ggml_tensor * features = inp_raw;
    ggml_tensor * qwen_features = nullptr;
    std::vector<ggml_tensor *> aux_levels;

    GGML_ASSERT(model.aux_conv_w.size() == 4);
    const int kernels[] = { 7, 3, 3, 3 };
    const int strides[] = { 4, 2, 2, 2 };
    const int paddings[] = { 3, 1, 1, 1 };
    const int dims[] = { 256, 512, 1024, 2048 };
    const int heads[] = { 8, 16, 32, 64 };
    const int groups[] = { 8, 16, 32, 64 };
    const int depths[] = { 1, 1, 9, 1 };
    int block_offset = 0;
    auto apply_conv_norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        if (!w) {
            return x;
        }
        const int64_t width = x->ne[0];
        const int64_t height = x->ne[1];
        const int64_t channels = x->ne[2];
        ggml_tensor * tokens = ggml_reshape_3d(ctx0, x, width * height, channels, 1);
        tokens = ggml_cont(ctx0, ggml_permute(ctx0, tokens, 1, 0, 2, 3));
        tokens = build_norm(tokens, w, b, NORM_TYPE_NORMAL, 1e-6f, -1);
        tokens = ggml_cont(ctx0, ggml_permute(ctx0, tokens, 1, 0, 2, 3));
        return ggml_cont(ctx0, ggml_reshape_4d(ctx0, tokens, width, height, channels, 1));
    };
    if (has_main_features) {
        const size_t n_values = encode_params->input_feature_map->size();
        GGML_ASSERT(n_values % 1280 == 0);
        const int width = encode_params->input_feature_map_width;
        const int height = encode_params->input_feature_map_height;
        GGML_ASSERT(width > 0 && height > 0 && (size_t) width * height * 1280 == n_values);
        features = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1280, width, height);
        ggml_set_name(features, "qwen2vl_feature_map");
        ggml_set_input(features);
        qwen_features = ggml_cont(ctx0, ggml_permute(ctx0, features, 2, 0, 1, 3));
        features = inp_raw;
    }

    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            features = apply_conv_norm(features, model.aux_conv_norm_w[i], model.aux_conv_norm_b[i]);
        }
        features = ggml_conv_2d(ctx0, model.aux_conv_w[i], features,
            kernels[i], kernels[i], paddings[i], paddings[i], strides[i], strides[i]);
        if (model.aux_conv_b[i]) {
            features = ggml_add(ctx0, features, ggml_reshape_4d(ctx0, model.aux_conv_b[i], 1, 1, dims[i], 1));
        }
        if (i == 0) {
            features = apply_conv_norm(features, model.aux_conv_norm_w[i], model.aux_conv_norm_b[i]);
        }

        const int width = features->ne[0];
        const int height = features->ne[1];
        features = ggml_permute(ctx0, features, 2, 0, 1, 3);
        features = ggml_cont_4d(ctx0, features, dims[i], width, height, 1);
        features = ggml_reshape_2d(ctx0, features, dims[i], width * height);

        for (int ib = 0; ib < depths[i]; ++ib) {
            const auto & layer = model.aux_davit_blocks[block_offset + ib];
            ggml_tensor * residual = features;
            features = davit_dwconv(ctx0, features, layer.spatial_dw1_w, layer.spatial_dw1_b,
                width, height, dims[i]);
            features = build_norm(features, layer.spatial_norm_w, layer.spatial_norm_b,
                NORM_TYPE_NORMAL, 1e-6f, -1);

            int nwin_x, nwin_y;
            ggml_tensor * windowed = davit_window_tokens(ctx0, features, width, height, dims[i], 12, nwin_x, nwin_y);
            const int window_tokens = 12 * 12;
            ggml_tensor * qkv = build_mm(layer.spatial_qkv_w, windowed);
            qkv = ggml_add(ctx0, qkv, layer.spatial_qkv_b);
            const int head_dim = dims[i] / heads[i];
            const size_t row_size = ggml_row_size(qkv->type, dims[i]);
            ggml_tensor * q = ggml_view_2d(ctx0, qkv, dims[i], window_tokens * nwin_x * nwin_y, qkv->nb[1], 0);
            ggml_tensor * k = ggml_view_2d(ctx0, qkv, dims[i], window_tokens * nwin_x * nwin_y, qkv->nb[1], row_size);
            ggml_tensor * v = ggml_view_2d(ctx0, qkv, dims[i], window_tokens * nwin_x * nwin_y, qkv->nb[1], row_size * 2);
            q = ggml_reshape_4d(ctx0, ggml_cont(ctx0, q), head_dim, heads[i], window_tokens, nwin_x * nwin_y);
            k = ggml_reshape_4d(ctx0, ggml_cont(ctx0, k), head_dim, heads[i], window_tokens, nwin_x * nwin_y);
            v = ggml_reshape_4d(ctx0, ggml_cont(ctx0, v), head_dim, heads[i], window_tokens, nwin_x * nwin_y);
            features = build_attn(layer.spatial_o_w, layer.spatial_o_b, q, k, v, nullptr,
                1.0f / sqrtf((float) head_dim), -1);
            features = davit_window_reverse(ctx0, features, width, height, dims[i], 12, nwin_x, nwin_y);
            features = davit_dwconv(ctx0, features, layer.spatial_dw2_w, layer.spatial_dw2_b,
                width, height, dims[i]);
            features = ggml_add(ctx0, features, residual);

            residual = features;
            features = build_norm(features, layer.spatial_ffn_norm_w, layer.spatial_ffn_norm_b,
                NORM_TYPE_NORMAL, 1e-6f, -1);
            features = build_ffn(features, layer.spatial_ffn_up_w, layer.spatial_ffn_up_b,
                nullptr, nullptr, layer.spatial_ffn_down_w, layer.spatial_ffn_down_b,
                FFN_GELU, -1);
            features = ggml_add(ctx0, features, residual);

            residual = features;
            features = davit_dwconv(ctx0, features, layer.channel_dw1_w, layer.channel_dw1_b,
                width, height, dims[i]);
            features = build_norm(features, layer.channel_norm_w, layer.channel_norm_b,
                NORM_TYPE_NORMAL, 1e-6f, -1);
            features = davit_channel_attn(this, layer, features, dims[i], groups[i], width * height);
            features = davit_dwconv(ctx0, features, layer.channel_dw2_w, layer.channel_dw2_b,
                width, height, dims[i]);
            features = ggml_add(ctx0, features, residual);
            residual = features;
            features = build_norm(features, layer.channel_ffn_norm_w, layer.channel_ffn_norm_b,
                NORM_TYPE_NORMAL, 1e-6f, -1);
            features = build_ffn(features, layer.channel_ffn_up_w, layer.channel_ffn_up_b,
                nullptr, nullptr, layer.channel_ffn_down_w, layer.channel_ffn_down_b,
                FFN_GELU, -1);
            features = ggml_add(ctx0, features, residual);


        }
        block_offset += depths[i];

        features = ggml_reshape_4d(ctx0, ggml_cont(ctx0, features), dims[i], width, height, 1);
        // ggml_permute maps each source axis to the destination axis.  The
        // token tensor is [C,W,H,B], while convolutions consume [W,H,C,B].
        features = ggml_permute(ctx0, features, 2, 0, 1, 3);
        features = ggml_cont(ctx0, features);
        aux_levels.push_back(features);
    }
    ggml_tensor * region;
    if (!has_main_features) {
        // The auxiliary GGUF can still be used for graph validation and
        // standalone experiments. Full HFRE requires the Qwen feature map.
        ggml_tensor * mean = ggml_sum(ctx0, features);
        region = ggml_repeat(ctx0, mean,
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 5888, n_tokens));
    } else {
        auto norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            return apply_conv_norm(x, w, b);
        };
        auto conv = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride) {
            ggml_tensor * y = stride == 1
                ? ggml_conv_2d(ctx0, w, x, 1, 1, 1, 1, 1, 1)
                : ggml_conv_transpose_2d_p0(ctx0, w, x, stride);
            return b ? ggml_add(ctx0, y, ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1)) : y;
        };
        std::vector<ggml_tensor *> fpn;
        GGML_ASSERT(qwen_features != nullptr && aux_levels.size() == 4);
        auto & l1 = model.aux_fpn_levels[0];
        auto & l2 = model.aux_fpn_levels[1];
        auto & l3 = model.aux_fpn_levels[2];
        auto & l4 = model.aux_fpn_levels[3];
        ggml_tensor * f1 = conv(qwen_features, l1.resize_w, l1.resize_b, 2);
        f1 = norm(f1, l1.resize_norm_w, l1.resize_norm_b);
        f1 = conv(f1, l1.resize2_w, l1.resize2_b, 2);
        f1 = norm(f1, l1.resize2_norm_w, l1.resize2_norm_b);
        f1 = norm(conv(f1, l1.proj_w, l1.proj_b, 1), l1.proj_norm_w, l1.proj_norm_b);
        fpn.push_back(norm(conv(f1, l1.out_w, l1.out_b, 1), l1.out_norm_w, l1.out_norm_b));
        ggml_tensor * f2 = conv(qwen_features, l2.resize_w, l2.resize_b, 2);
        f2 = norm(f2, l2.resize_norm_w, l2.resize_norm_b);
        fpn.push_back(norm(conv(f2, l2.proj_w, l2.proj_b, 1), l2.proj_norm_w, l2.proj_norm_b));
        fpn.push_back(norm(conv(qwen_features, l3.proj_w, l3.proj_b, 1), l3.proj_norm_w, l3.proj_norm_b));
        fpn.push_back(norm(conv(qwen_features, l4.proj_w, l4.proj_b, 1), l4.proj_norm_w, l4.proj_norm_b));
        const int tw = fpn[0]->ne[0], th = fpn[0]->ne[1];
        for (auto & f : fpn) {
            f = ggml_interpolate(ctx0, f, tw, th, f->ne[2], f->ne[3],
                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
        }
        ggml_tensor * semantic_map = fpn[0];
        for (size_t i = 1; i < fpn.size(); ++i) semantic_map = ggml_concat(ctx0, semantic_map, fpn[i], 2);
        semantic_map = ggml_cont(ctx0, ggml_permute(ctx0, semantic_map, 1, 2, 0, 3));
        ggml_tensor * boxes = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 4, n_tokens);
        ggml_set_name(boxes, "bbox");
        ggml_set_input(boxes);
        ggml_tensor * bbox_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 5888, n_tokens);
        ggml_set_name(bbox_pos, "bbox_pos");
        ggml_set_input(bbox_pos);
        ggml_tensor * pooled = roi_align(ctx0, semantic_map, boxes, tw, th, 14, n_tokens,
            (float) tw / img.nx(), (float) th / img.ny());
        pooled = ggml_reshape_3d(ctx0, pooled, 2048, 14 * 14, n_tokens);
        pooled = ggml_cont(ctx0, ggml_permute(ctx0, pooled, 1, 0, 2, 3));
        ggml_tensor * semantic = ggml_cont_2d(ctx0, ggml_mean(ctx0, pooled), 2048, n_tokens);

        const int aw = aux_levels[0]->ne[0];
        const int ah = aux_levels[0]->ne[1];
        ggml_tensor * perception_map = aux_levels[0];
        for (size_t i = 1; i < aux_levels.size(); ++i) {
            ggml_tensor * level = aux_levels[i];
            level = ggml_interpolate(ctx0, level, aw, ah, level->ne[2], level->ne[3],
                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
            perception_map = ggml_concat(ctx0, perception_map, level, 2);
        }
        perception_map = ggml_cont(ctx0, ggml_permute(ctx0, perception_map, 1, 2, 0, 3));
        ggml_tensor * perception = roi_align(ctx0, perception_map, boxes, aw, ah, 14, n_tokens,
            0.25f, 0.25f);
        perception = ggml_reshape_3d(ctx0, perception, 3840, 14 * 14, n_tokens);
        perception = ggml_cont(ctx0, ggml_permute(ctx0, perception, 1, 0, 2, 3));
        perception = ggml_cont_2d(ctx0, ggml_mean(ctx0, perception), 3840, n_tokens);
        region = ggml_concat(ctx0, perception, semantic, 0);
        region = ggml_add(ctx0, region, bbox_pos);
    }
    ggml_tensor * embeddings = build_ffn(region,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_1_w, model.mm_1_b,
        FFN_GELU,
        -1);
    ggml_set_name(embeddings, "vlm_fo1_aux_embeddings");
    ggml_build_forward_expand(gf, embeddings);
    return gf;
}