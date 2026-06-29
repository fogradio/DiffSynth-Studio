"""WISA-80K dataset operator for Wan2.1 T2V joint-copilot training.

Design constraints (agreed with the user, see also WISA-80K_使用说明.md):

  * By default, each clip yields
    ``num_frames = align_4k1(min(max_frames, total_frames))`` so it satisfies
    Wan's ``4k+1`` (VAE 4x temporal compression) requirement and caps at 81
    frames (~5s @ 16fps). In fixed-frame mode (used for batch_size > 1), short
    clips repeat sampled timestamps so every sample has the same frame count.
  * Frames uniformly cover the *whole* clip via ``round(linspace(0, N-1, n))``.
    In the default mode, indices are strictly increasing; fixed-frame mode may
    repeat indices for short clips. Both modes keep the full physical process
    in view (critical for WISA) instead of only the first 5s.
  * Prompt = ``captions`` + ``phys_law`` (physical-consistency level "B").
  * Spatial size is unified to (height, width) by ``ImageCropAndResize``.
"""

import os

import imageio
import torch
from PIL import Image


def align_frame_count(n, factor=4, remainder=1):
    """Largest m <= n with ``m % factor == remainder`` (and m >= 1)."""
    m = int(n)
    while m > 1 and m % factor != remainder:
        m -= 1
    return max(m, 1)


def build_wisa_prompt(item, level="B"):
    """captions (+ phys_law for level B/C, + qualitative phenomena for C)."""
    cap = (item.get("captions") or "").strip()
    if level == "A":
        return cap
    pa = item.get("physical_annotation") or {}
    phys = (pa.get("phys_law") or "").strip()
    text = cap if not phys else f"{cap}\n\nPhysical considerations: {phys}"
    if level == "C":
        phen = [
            pa[q]
            for q in ("q0", "q1", "q2")
            if pa.get(q) and "no obvious" not in str(pa[q]).lower()
        ]
        if phen:
            text += "\nPhysical phenomena: " + "; ".join(p.strip() for p in phen)
    return text


def build_video_index(video_root):
    """Map ``video_name`` (``<sha256>.mp4``) -> absolute path.

    The same hash can appear under several shard dirs (0..127); ``setdefault``
    keeps the first occurrence, which de-duplicates exactly like the dataset
    doc recommends (always resolve via JSON ``video_name``, not a raw glob).
    """
    name2path = {}
    for entry in sorted(os.listdir(video_root)):
        sub = os.path.join(video_root, entry)
        if not os.path.isdir(sub):
            continue
        for fn in os.listdir(sub):
            if fn.endswith(".mp4"):
                name2path.setdefault(fn, os.path.join(sub, fn))
    return name2path


class WISAVideoOperator:
    def __init__(
        self,
        name2path,
        frame_processor,
        max_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        prompt_level="B",
        fixed_frames=False,
    ):
        self.name2path = name2path
        self.frame_processor = frame_processor
        self.max_frames = max_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.prompt_level = prompt_level
        # When True every clip yields a CONSTANT align_4k1(max_frames) count so
        # samples can be stacked into one [B, C, T, H, W] batch (single-GPU
        # batch_size > 1). Clips longer than the target stay full-span sparse;
        # clips shorter than it reuse frames via round(linspace) — i.e. some
        # timestamps freeze (中间静默 padding) instead of dropping the sample
        # or zero-padding the tensor.
        self.fixed_frames = fixed_frames

    def _resolve_path(self, item):
        path = item.get("video_path") or self.name2path.get(item["video_name"])
        if path is None:
            raise KeyError(
                f"No video file for {item['video_name']!r} in the WISA video index."
            )
        return path

    def _sample_indices(self, total_frames):
        if self.fixed_frames:
            # Constant length across clips (batched training). Do NOT min() with
            # total_frames: short clips repeat frames via the round(linspace)
            # below rather than yielding a smaller tensor.
            num_frames = align_frame_count(
                self.max_frames,
                self.time_division_factor,
                self.time_division_remainder,
            )
        else:
            num_frames = align_frame_count(
                min(self.max_frames, total_frames),
                self.time_division_factor,
                self.time_division_remainder,
            )
        if num_frames <= 1:
            return [0]
        step = (total_frames - 1) / (num_frames - 1)
        return [int(round(i * step)) for i in range(num_frames)]

    def __call__(self, item):
        item = item.copy()
        path = self._resolve_path(item)
        reader = imageio.get_reader(path)
        try:
            total_frames = reader.count_frames()
            indices = self._sample_indices(total_frames)
            frames = []
            for idx in indices:
                frame = Image.fromarray(reader.get_data(idx))
                frames.append(self.frame_processor(frame))
        finally:
            reader.close()
        item["video"] = frames
        item["prompt"] = build_wisa_prompt(item, self.prompt_level)
        # Give the mistake-forcing recorder a stable per-sample id for logging.
        item.setdefault("sample_id", item.get("video_name"))
        return item


class WISAManifestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_operator):
        self.dataset = dataset
        self.sample_operator = sample_operator
        self.load_from_cache = dataset.load_from_cache

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.sample_operator(self.dataset[index])
