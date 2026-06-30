"""Inference script: Wan2.1-T2V-1.3B with Video Copilot correction.

At each denoising step the base DiT predicts a velocity field.  Before applying
CFG, the copilot model receives the estimated clean prediction (x0) together
with the DiT's time embedding, condition embedding and hidden-state mean, and
outputs a velocity residual.  Adding this residual back yields a corrected
velocity that is then used in the standard flow-matching ODE step.

Supports copilot v1 (VideoCopilotDecoder), v2 (VideoCopilotDecoderV2) and v3
(VideoCopilotDecoderV3).  v1/v2 consume the layer-averaged hidden mean; v3
consumes the four selected DiT hidden states fed through its cross-attention.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure DiffSynth-Studio is on the path (the shell wrapper already does this
# via PYTHONPATH, but be safe).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # DiffSynth-Studio root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# We reuse the hidden-state capture helper already used by the training code.
MISTAKE_FORCING_DIR = Path(__file__).resolve().parents[4]  # mistake_forcing root
sys.path.insert(0, str(MISTAKE_FORCING_DIR / "DiffSynth-Studio" / "examples" / "wanvideo" / "model_training"))
from mistake_forcing import WanHiddenStateCapture

# ---------------------------------------------------------------------------
# Copilot model loader
# ---------------------------------------------------------------------------

COPILOT_ROOT = MISTAKE_FORCING_DIR / "video_copilot"
sys.path.insert(0, str(COPILOT_ROOT))


def _infer_v2_depth_from_state_dict(state_dict) -> int | None:
    """Look at `blocks.<idx>.*` keys to figure out how many decoder blocks
    the checkpoint was trained with. Returns None if no blocks key is found."""
    max_idx = -1
    for k in state_dict:
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
    return max_idx + 1 if max_idx >= 0 else None


def load_copilot_model(
    variant: str,
    ckpt_path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    **override_kwargs,
) -> torch.nn.Module:
    """Instantiate a copilot model and load weights from *safetensors* or *.pt*.

    Resolution order for model hyper-params (dim/depth/num_heads/mlp_ratio):
      1. explicit ``override_kwargs`` from the caller (highest priority)
      2. ``model_config.json`` saved alongside the checkpoint by train.py
      3. for v2: depth inferred from the highest ``blocks.<i>.*`` index in the
         state_dict (handles older ckpts that have no model_config.json)
      4. hard-coded defaults (lowest priority)
    """
    if variant == "v1":
        from model import VideoCopilotDecoder as CopilotCls
        defaults = dict(dim=1024, depth=8, num_heads=16, mlp_ratio=4.0)
    elif variant == "v2":
        from model_v2 import VideoCopilotDecoderV2 as CopilotCls
        # Default depth matches video_copilot/model_v2.py (Wan-style AdaLN, depth=10).
        defaults = dict(dim=1024, depth=10, num_heads=16, mlp_ratio=4.0)
    elif variant == "v3":
        from model_v3 import VideoCopilotDecoderV3 as CopilotCls
        # Same Wan-style AdaLN trunk as v2 (depth=10); the memory side fuses the
        # four selected DiT hidden states. n_selected_layers/selected_hidden_dim
        # fall back to the model's own defaults (4 / 1536), matching training.
        defaults = dict(dim=1024, depth=10, num_heads=16, mlp_ratio=4.0)
    else:
        raise ValueError(f"Unknown copilot variant: {variant}")

    # Load state dict first so we can both inspect it and feed it into the model.
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path, device="cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    # (2) Try model_config.json sitting next to the ckpt — train.py writes this.
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cfg_path = os.path.join(ckpt_dir, "model_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        for key in ("dim", "depth", "num_heads", "mlp_ratio"):
            if key in cfg:
                defaults[key] = cfg[key]
        print(f"[copilot] loaded hyper-params from {cfg_path}: "
              f"dim={defaults['dim']} depth={defaults['depth']} "
              f"num_heads={defaults['num_heads']} mlp_ratio={defaults['mlp_ratio']}")
    elif variant == "v2":
        # (3) Fallback: infer v2 depth from the state_dict block indices.
        inferred = _infer_v2_depth_from_state_dict(state_dict)
        if inferred is not None and inferred != defaults["depth"]:
            print(f"[copilot] inferred depth={inferred} from ckpt keys "
                  f"(default was {defaults['depth']})")
            defaults["depth"] = inferred

    # (1) Caller overrides always win.
    defaults.update(override_kwargs)
    model = CopilotCls(**defaults)

    # Filter out legacy keys (e.g. pos_h/pos_t/pos_w from older v1 ckpts, or
    # time_proj / 3-row seg_embed / per-block adaLN.* / final_adaLN.* / time_mlp.*
    # from intermediate v2 variants before Wan-style AdaLN landed).
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    skipped = set(state_dict.keys()) - model_keys
    if skipped:
        print(f"[copilot] skipping {len(skipped)} legacy key(s): {sorted(skipped)}")
    model.load_state_dict(filtered, strict=True)
    model.eval()
    model.to(device=device, dtype=dtype)
    print(f"[copilot] loaded {variant} from {ckpt_path}  "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    return model


# ---------------------------------------------------------------------------
# Default prompts / negative prompt
# ---------------------------------------------------------------------------

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wan2.1-T2V-1.3B inference with Video Copilot correction",
    )
    # ---- prompts / IO ----
    p.add_argument("--prompts_path", type=str,
                    default=str(MISTAKE_FORCING_DIR / "inference_prompts.jsonl"))
    p.add_argument("--base_model_root", type=str,
                    default="/mnt/workspace/common/models/Wan2.1-T2V-1.3B")
    p.add_argument("--dit_weights", type=str,
                    default="/mnt/workspace/common/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    p.add_argument("--output_root", type=str,
                    default=str(MISTAKE_FORCING_DIR / "outputs" / "inference"))
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--tag", type=str, default="copilot")
    p.add_argument("--limit", type=int, default=0)

    # ---- copilot ----
    p.add_argument("--copilot_variant", type=str, default="v2", choices=["v1", "v2", "v3"],
                    help="Copilot model architecture variant (v3 consumes the selected DiT hidden states).")
    p.add_argument("--copilot_ckpt", type=str,
                    default=str(MISTAKE_FORCING_DIR / "outputs" / "video_copilot_v2" / "epoch-17" / "model.safetensors"),
                    help="Path to copilot model weights (safetensors or pt).")
    p.add_argument("--copilot_scale", type=float, default=1.0,
                    help="Multiplier for the copilot velocity residual (0 = disabled).")
    p.add_argument("--copilot_start_pct", type=float, default=0.0,
                    help="Start applying copilot after this fraction of denoising steps (0.0 = from the very start).")
    p.add_argument("--copilot_end_pct", type=float, default=1.0,
                    help="Stop applying copilot after this fraction of denoising steps (1.0 = until the very end).")

    # ---- generation ----
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def coerce_prompt_text(prompt) -> str:
    # Prompt values may be a dict (e.g. PhyGenBench entries kept un-split); the
    # pipeline only accepts a string, so serialize non-str prompts to JSON.
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, ensure_ascii=False)


def build_output_dir(output_root: str, tag: str) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(output_root) / f"{timestamp}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Core: generate one video with copilot correction
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_copilot(
    pipe: WanVideoPipeline,
    copilot: torch.nn.Module,
    *,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    cfg_scale: float,
    sigma_shift: float,
    seed: int,
    copilot_variant: str = "v2",
    copilot_scale: float = 1.0,
    copilot_start_pct: float = 0.0,
    copilot_end_pct: float = 1.0,
) -> list:
    """Run the full Wan2.1 T2V pipeline with copilot correction in the loop.

    This replicates :meth:`WanVideoPipeline.__call__` but injects the copilot
    velocity-residual correction at every (eligible) denoising step.  The
    correction is applied to the *positive* velocity prediction, *before* CFG
    combination, so that it matches the copilot's training distribution.
    """

    # 1. Scheduler -------------------------------------------------------
    pipe.scheduler.set_timesteps(
        num_inference_steps, denoising_strength=1.0, shift=sigma_shift)

    # 2. Prepare input dicts (same as pipeline.__call__) -----------------
    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": negative_prompt}
    inputs_shared = {
        "input_image": None, "end_image": None,
        "input_video": None, "denoising_strength": 1.0,
        "control_video": None, "reference_image": None,
        "camera_control_direction": None, "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5,
                                  0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None, "vace_video_mask": None,
        "vace_reference_image": None, "vace_scale": 1.0,
        "seed": seed, "rand_device": "cpu",
        "height": height, "width": width, "num_frames": num_frames,
        "cfg_scale": cfg_scale, "cfg_merge": False,
        "sigma_shift": sigma_shift,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": True, "tile_size": (30, 52), "tile_stride": (15, 26),
        "sliding_window_size": None, "sliding_window_stride": None,
        "input_audio": None, "audio_sample_rate": 16000,
        "s2v_pose_video": None, "audio_embeds": None,
        "s2v_pose_latents": None, "motion_video": None,
        "animate_pose_video": None, "animate_face_video": None,
        "animate_inpaint_video": None, "animate_mask_video": None,
        "vap_video": None,
        "wantodance_music_path": None, "wantodance_reference_image": None,
        "wantodance_fps": 30, "wantodance_keyframes": None,
        "wantodance_keyframes_mask": None,
        "framewise_decoding": False,
    }

    # These keys are expected by some units but not relevant for T2V-1.3B.
    inputs_posi.setdefault("tea_cache_l1_thresh", None)
    inputs_posi.setdefault("tea_cache_model_id", "")
    inputs_posi.setdefault("num_inference_steps", num_inference_steps)
    inputs_nega.setdefault("tea_cache_l1_thresh", None)
    inputs_nega.setdefault("tea_cache_model_id", "")
    inputs_nega.setdefault("num_inference_steps", num_inference_steps)
    inputs_nega.setdefault("negative_vap_prompt", " ")
    inputs_posi.setdefault("vap_prompt", " ")

    # 3. Run pre-processing units ----------------------------------------
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    # 4. Load models to device -------------------------------------------
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    # 5. Determine copilot step range ------------------------------------
    total_steps = len(pipe.scheduler.timesteps)
    copilot_start_step = int(copilot_start_pct * total_steps)
    copilot_end_step = int(copilot_end_pct * total_steps)

    # 6. Denoising loop with copilot correction --------------------------
    for progress_id, timestep in enumerate(tqdm(
            pipe.scheduler.timesteps, desc="Denoising")):
        timestep_t = timestep.unsqueeze(0).to(
            dtype=pipe.torch_dtype, device=pipe.device)

        # -- Decide whether to apply copilot at this step ----------------
        apply_copilot = (
            copilot_scale != 0.0
            and copilot_start_step <= progress_id < copilot_end_step
        )

        # -- Positive-prompt forward pass --------------------------------
        if apply_copilot:
            capture = WanHiddenStateCapture()
            noise_pred_posi = pipe.model_fn(
                **models, **inputs_shared, **inputs_posi,
                timestep=timestep_t, mistake_capture=capture,
            )
        else:
            noise_pred_posi = pipe.model_fn(
                **models, **inputs_shared, **inputs_posi,
                timestep=timestep_t,
            )

        # -- Copilot correction (before CFG) -----------------------------
        if apply_copilot:
            # Compute the estimated clean prediction (step to sigma=0)
            clean_prediction = pipe.scheduler.step(
                noise_pred_posi,
                pipe.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
                to_final=True,
            )
            # Extract captured states from the positive forward pass
            capture_payload = capture.finalize()
            time_embedding = capture_payload["time_embedding"]   # (B, dim)
            condition_embedding = inputs_posi["context"]         # (B, S, text_dim)

            # v3 fuses the four selected DiT hidden states; v1/v2 use the
            # layer-averaged hidden mean. Match the training-time `write()` path.
            if copilot_variant == "v3":
                # finalize() stacks layers on dim 0 -> (L, B, N, D);
                # v3 expects (B, L, N, D).
                hidden_input = capture_payload["selected_hidden_states"]
                if hidden_input.dim() == 4:
                    hidden_input = hidden_input.permute(1, 0, 2, 3).contiguous()
            else:
                hidden_input = capture_payload["hidden_mean"]    # (B, N, dim)

            # Run copilot
            copilot_dtype = next(copilot.parameters()).dtype
            velocity_residual = copilot(
                clean_prediction.to(copilot_dtype),
                time_embedding.to(copilot_dtype),
                condition_embedding.to(copilot_dtype),
                hidden_input.to(copilot_dtype),
            )
            # Correct positive velocity
            noise_pred_posi = (
                noise_pred_posi
                + copilot_scale * velocity_residual.to(noise_pred_posi.dtype)
            )

        # -- CFG ---------------------------------------------------------
        if cfg_scale != 1.0:
            noise_pred_nega = pipe.model_fn(
                **models, **inputs_shared, **inputs_nega,
                timestep=timestep_t,
            )
            noise_pred = (
                noise_pred_nega
                + cfg_scale * (noise_pred_posi - noise_pred_nega)
            )
        else:
            noise_pred = noise_pred_posi

        # -- Scheduler step ----------------------------------------------
        inputs_shared["latents"] = pipe.scheduler.step(
            noise_pred,
            pipe.scheduler.timesteps[progress_id],
            inputs_shared["latents"],
        )

    # 7. Post-processing -------------------------------------------------
    for unit in pipe.post_units:
        inputs_shared, _, _ = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    # 8. VAE decode ------------------------------------------------------
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        inputs_shared["latents"], device=pipe.device,
        tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
    video = pipe.vae_output_to_video(video)
    pipe.load_models_to_device([])
    return video


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Resolve paths ----
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

    # ---- Print config ----
    print(f"[info] output dir:       {out_dir}")
    print(f"[info] DiT weights:      {dit_path}")
    print(f"[info] copilot variant:  {args.copilot_variant}")
    print(f"[info] copilot ckpt:     {args.copilot_ckpt}")
    print(f"[info] copilot scale:    {args.copilot_scale}")
    print(f"[info] copilot range:    [{args.copilot_start_pct:.0%}, {args.copilot_end_pct:.0%})")
    print(f"[info] prompts:          {args.prompts_path} (n={len(prompts)})")

    # ---- Load pipeline ----
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

    # ---- Load copilot ----
    copilot = load_copilot_model(
        variant=args.copilot_variant,
        ckpt_path=args.copilot_ckpt,
        device=args.device,
        dtype=torch.bfloat16,
    )

    # ---- Generate ----
    run_meta = {
        "args": vars(args),
        "dit_weights_resolved": dit_path,
        "base_model_root": str(base_root),
        "copilot_variant": args.copilot_variant,
        "copilot_ckpt": args.copilot_ckpt,
        "copilot_scale": args.copilot_scale,
        "copilot_start_pct": args.copilot_start_pct,
        "copilot_end_pct": args.copilot_end_pct,
        "negative_prompt": args.negative_prompt,
        "num_prompts": len(prompts),
        "items": [],
    }

    for idx, rec in enumerate(prompts):
        prompt = coerce_prompt_text(rec["prompt"])
        sample_id = rec.get("sample_id", f"prompt_{idx:03d}")
        safe_id = sample_id.replace("/", "_")
        video_name = f"{idx:02d}_{safe_id}.mp4"
        video_path = out_dir / video_name

        print(f"\n[gen {idx + 1}/{len(prompts)}] {sample_id}")
        print(f"        prompt: {prompt[:140]}{'...' if len(prompt) > 140 else ''}")

        seed = args.seed + idx
        video = generate_with_copilot(
            pipe, copilot,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            seed=seed,
            copilot_variant=args.copilot_variant,
            copilot_scale=args.copilot_scale,
            copilot_start_pct=args.copilot_start_pct,
            copilot_end_pct=args.copilot_end_pct,
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
