#!/usr/bin/env python3
"""Export the VLM-FO1 auxiliary region encoder to a standalone GGUF file.

The regular text and mmproj converters intentionally do not consume this
branch: it is a second image encoder plus the HFRE/region projector, not a
standard CLIP projector.  Keeping it in its own file also lets the runtime
loader evolve without invalidating existing Qwen2.5-VL mmproj files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

# Allow running this file directly from a source checkout, like the other
# conversion entry points.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))

from gguf import GGUFWriter


AUX_PREFIXES = (
    "model.vision_tower_aux.",
    "model.object_vp_extractor.",
    "model.mm_projector_aux.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Hugging Face model directory")
    parser.add_argument("-o", "--output", type=Path, help="Output auxiliary GGUF path")
    parser.add_argument(
        "--dtype",
        choices=("f16", "f32"),
        default="f16",
        help="Storage dtype for matrix weights (default: f16)",
    )
    return parser.parse_args()


def load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    with index_path.open(encoding="utf-8") as file:
        weight_map = json.load(file)["weight_map"]
    return {
        name: filename
        for name, filename in weight_map.items()
        if name.startswith(AUX_PREFIXES)
    }


def gguf_name(name: str) -> str:
    if name.startswith("model."):
        name = name[len("model.") :]
    name = name.replace("vision_tower_aux.image_tower", "fo1.vt")
    name = name.replace("object_vp_extractor.simple_fpn", "fo1.fpn")
    name = name.replace("mm_projector_aux", "fo1.proj")
    if name.startswith("fo1.vt.blocks."):
        parts = name.split(".")
        stage, block = parts[3], parts[4]
        suffix = ".".join(parts[5:])
        short = {
            "spatial_block.window_attn.norm.weight": "snw",
            "spatial_block.window_attn.norm.bias": "snb",
            "spatial_block.window_attn.fn.qkv.weight": "sqw",
            "spatial_block.window_attn.fn.qkv.bias": "sqb",
            "spatial_block.window_attn.fn.proj.weight": "sow",
            "spatial_block.window_attn.fn.proj.bias": "sob",
            "spatial_block.conv1.fn.dw.weight": "sd1w",
            "spatial_block.conv1.fn.dw.bias": "sd1b",
            "spatial_block.conv2.fn.dw.weight": "sd2w",
            "spatial_block.conv2.fn.dw.bias": "sd2b",
            "spatial_block.ffn.norm.weight": "sfnw",
            "spatial_block.ffn.norm.bias": "sfnb",
            "spatial_block.ffn.fn.net.fc1.weight": "sfuw",
            "spatial_block.ffn.fn.net.fc1.bias": "sfub",
            "spatial_block.ffn.fn.net.fc2.weight": "sfdw",
            "spatial_block.ffn.fn.net.fc2.bias": "sfdb",
            "channel_block.channel_attn.norm.weight": "cnw",
            "channel_block.channel_attn.norm.bias": "cnb",
            "channel_block.channel_attn.fn.qkv.weight": "cqw",
            "channel_block.channel_attn.fn.qkv.bias": "cqb",
            "channel_block.channel_attn.fn.proj.weight": "cow",
            "channel_block.channel_attn.fn.proj.bias": "cob",
            "channel_block.conv1.fn.dw.weight": "cd1w",
            "channel_block.conv1.fn.dw.bias": "cd1b",
            "channel_block.conv2.fn.dw.weight": "cd2w",
            "channel_block.conv2.fn.dw.bias": "cd2b",
            "channel_block.ffn.norm.weight": "cfnw",
            "channel_block.ffn.norm.bias": "cfnb",
            "channel_block.ffn.fn.net.fc1.weight": "cfuw",
            "channel_block.ffn.fn.net.fc1.bias": "cfub",
            "channel_block.ffn.fn.net.fc2.weight": "cfdw",
            "channel_block.ffn.fn.net.fc2.bias": "cfdb",
        }.get(suffix)
        if short is not None:
            return f"fo1.b{stage}.{block}.{short}"
    return name


def tensor_dtype(name: str, data: np.ndarray, requested: str) -> np.ndarray:
    # Keep normalization/statistics and biases exact.  This mirrors the
    # usual llama.cpp converter policy and avoids losing small constants.
    if requested == "f16" and data.ndim >= 2 and name.endswith(".weight"):
        return data.astype(np.float16, copy=False)
    return data.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output = args.output or model_dir / "vlm-fo1-aux-f16.gguf"
    weights = load_index(model_dir)
    if not weights:
        raise ValueError("no VLM-FO1 auxiliary tensors found in the safetensors index")

    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(path=str(output), arch="clip")
    writer.add_name("VLM-FO1 auxiliary region encoder")
    writer.add_description("DaViT auxiliary tower and HFRE region encoder for VLM-FO1")
    writer.add_file_type(1 if args.dtype == "f16" else 0)
    writer.add_clip_has_vision_encoder(True)
    writer.add_string("clip.projector_type", "vlm_fo1_aux")
    writer.add_uint32("clip.vision.embedding_length", 2048)
    writer.add_uint32("clip.vision.feed_forward_length", 8192)
    writer.add_uint32("clip.vision.block_count", 1)
    writer.add_uint32("clip.vision.attention.head_count", 16)
    writer.add_uint32("clip.vision.attention.head_dim", 128)
    writer.add_uint32("clip.vision.projection_dim", 2048)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_uint32("clip.vision.image_size", 1024)
    writer.add_uint32("clip.vision.patch_size", 14)
    writer.add_uint32("clip.vision.image_min_pixels", 256 * 28 * 28)
    writer.add_uint32("clip.vision.image_max_pixels", 1024 * 1024)
    writer.add_array("clip.vision.image_mean", [0.485, 0.456, 0.406])
    writer.add_array("clip.vision.image_std", [0.229, 0.224, 0.225])
    writer.add_string("vlm_fo1.model_type", config.get("model_type", "omchat_qwen2_5_vl"))
    writer.add_string("vlm_fo1.mm_vision_tower_aux", config.get("mm_vision_tower_aux", ""))
    writer.add_string("vlm_fo1.mm_projector_aux_type", config.get("mm_projector_aux_type", ""))
    writer.add_uint32("vlm_fo1.num_region_tokens", int(config.get("mm_num_region_tokens", 100)))
    writer.add_array("vlm_fo1.davit.depths", [1, 1, 9, 1])
    writer.add_array("vlm_fo1.davit.embed_dims", [256, 512, 1024, 2048])
    writer.add_array("vlm_fo1.davit.num_heads", [8, 16, 32, 64])
    writer.add_array("vlm_fo1.davit.num_groups", [8, 16, 32, 64])
    writer.add_array("vlm_fo1.davit.patch_size", [7, 3, 3, 3])
    writer.add_array("vlm_fo1.davit.patch_stride", [4, 2, 2, 2])
    writer.add_array("vlm_fo1.davit.patch_padding", [3, 1, 1, 1])
    writer.add_uint32("vlm_fo1.davit.window_size", 12)
    writer.add_uint32("vlm_fo1.davit.mlp_ratio", 4)

    for source_name, filename in sorted(weights.items()):
        with safe_open(str(model_dir / filename), framework="pt") as handle:
            tensor = handle.get_tensor(source_name).detach().cpu()
        data = tensor_dtype(source_name, tensor.float().numpy(), args.dtype)
        name = gguf_name(source_name)
        writer.add_tensor(name, data)
        print(f"{name}: {data.dtype} {tuple(data.shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {len(weights)} tensors to {output}")


if __name__ == "__main__":
    main()