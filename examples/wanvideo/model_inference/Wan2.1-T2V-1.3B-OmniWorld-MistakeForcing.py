"""Inference script: Wan2.1-T2V-1.3B with OmniWorld MistakeForcing SFT weights.

Reads prompts from a JSONL (one JSON object per line, each with a `prompt` field
and optional `sample_id`), runs text-to-video generation, and writes mp4 files
plus a metadata JSON into a timestamped output folder.

The DiT weights default to the base model's `diffusion_pytorch_model.safetensors`
but can be replaced by any SFT checkpoint (e.g. epoch-4.safetensors) via the
`--dit_weights` flag. T5 / VAE / tokenizer always come from the base model dir.
"""
import argparse
import datetime as _dt
import json
import os
from pathlib import Path

import torch

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.1-T2V-1.3B inference with OmniWorld MistakeForcing weights",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="/mnt/workspace/hwzhang/code/mistake_forcing/inference_prompts.jsonl",
        help="JSONL file containing prompts (`prompt` field per line).",
    )
    parser.add_argument(
        "--base_model_root",
        type=str,
        default="/mnt/workspace/common/models/Wan2.1-T2V-1.3B",
        help="Base Wan2.1-T2V-1.3B model directory (provides T5 / VAE / tokenizer, "
             "and default DiT weights).",
    )
    parser.add_argument(
        "--dit_weights",
        type=str,
        default="/mnt/workspace/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_omniworld_sft_mistake/epoch-4.safetensors",
        help="Path to DiT weights. Set empty string '' to use the base model's "
             "diffusion_pytorch_model.safetensors.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/mnt/workspace/hwzhang/code/mistake_forcing/outputs/inference",
        help="Parent directory; a timestamped subfolder will be created inside.",
    )
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device for the pipeline (e.g. 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="omniworld_mistake",
        help="Short tag appended to the timestamped output folder name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only run the first N prompts (handy for smoke tests).",
    )
    return parser.parse_args()


def resolve_dit_path(base_root: str, dit_weights: str) -> str:
    if dit_weights:
        path = Path(dit_weights)
        if not path.is_file():
            raise FileNotFoundError(f"DiT weights not found: {path}")
        return str(path)
    base_dit = Path(base_root) / "diffusion_pytorch_model.safetensors"
    if not base_dit.is_file():
        raise FileNotFoundError(f"Base DiT weights not found: {base_dit}")
    return str(base_dit)


def load_prompts(prompts_path: str) -> list[dict]:
    records: list[dict] = []
    with open(prompts_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "prompt" not in rec or not rec["prompt"]:
                raise ValueError(f"Line {i}: missing non-empty `prompt` field")
            records.append(rec)
    if not records:
        raise ValueError(f"No prompts found in {prompts_path}")
    return records


def build_output_dir(output_root: str, tag: str) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(output_root) / f"{timestamp}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    args = parse_args()

    dit_path = resolve_dit_path(args.base_model_root, args.dit_weights)
    base_root = Path(args.base_model_root)
    t5_path = base_root / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = base_root / "Wan2.1_VAE.pth"
    tokenizer_dir = base_root / "google" / "umt5-xxl"
    for p in (t5_path, vae_path, tokenizer_dir):
        if not p.exists():
            raise FileNotFoundError(f"Required base model asset missing: {p}")

    prompts = load_prompts(args.prompts_path)
    if args.limit and args.limit > 0:
        prompts = prompts[: args.limit]

    out_dir = build_output_dir(args.output_root, args.tag)
    print(f"[info] output dir: {out_dir}")
    print(f"[info] DiT weights: {dit_path}")
    print(f"[info] T5 weights:  {t5_path}")
    print(f"[info] VAE weights: {vae_path}")
    print(f"[info] tokenizer:   {tokenizer_dir}")
    print(f"[info] prompts:     {args.prompts_path} (n={len(prompts)})")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=dit_path),
            ModelConfig(path=str(t5_path)),
            ModelConfig(path=str(vae_path)),
        ],
        tokenizer_config=ModelConfig(path=str(tokenizer_dir)),
    )

    run_meta = {
        "args": vars(args),
        "dit_weights_resolved": dit_path,
        "base_model_root": str(base_root),
        "negative_prompt": args.negative_prompt,
        "num_prompts": len(prompts),
        "items": [],
    }

    for idx, rec in enumerate(prompts):
        prompt = rec["prompt"]
        sample_id = rec.get("sample_id", f"prompt_{idx:03d}")
        safe_id = sample_id.replace("/", "_")
        video_name = f"{idx:02d}_{safe_id}.mp4"
        video_path = out_dir / video_name

        print(f"\n[gen {idx + 1}/{len(prompts)}] {sample_id}")
        print(f"        prompt: {prompt[:140]}{'...' if len(prompt) > 140 else ''}")

        seed = args.seed + idx
        video = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            seed=seed,
            tiled=True,
        )
        save_video(video, str(video_path), fps=args.fps, quality=5)
        print(f"        saved -> {video_path}")

        run_meta["items"].append({
            "index": idx,
            "sample_id": sample_id,
            "prompt": prompt,
            "seed": seed,
            "video_path": str(video_path),
        })

    meta_path = out_dir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"\n[done] wrote {len(prompts)} videos + metadata to {out_dir}")


if __name__ == "__main__":
    main()
