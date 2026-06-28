import json
import re

import torch
from PIL import Image

from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion.loss import FlowMatchSFTMistakeForcingLoss

from examples.wanvideo.model_training.mistake_forcing import (
    OmniWorldFrameSequenceOperator,
    WanHiddenStateCapture,
    WanMistakeRecorder,
)


def test_omniworld_operator_loads_video_and_prompt(tmp_path):
    color_dir = tmp_path / "scene_a" / "color"
    color_dir.mkdir(parents=True)
    for frame_index in range(81):
        image = Image.new("RGB", (960, 540), color=(frame_index % 255, 12, 34))
        image.save(color_dir / f"{frame_index:06d}.png")

    text_dir = tmp_path / "scene_a" / "text"
    text_dir.mkdir(parents=True)
    text_path = text_dir / "000001_000081.json"
    text_path.write_text(json.dumps({"captions": {"Video_Caption": "prompt from json"}}), encoding="utf-8")

    operator = OmniWorldFrameSequenceOperator(
        base_path=str(tmp_path),
        frame_processor=ImageCropAndResize(480, 832, None, 16, 16),
        num_frames=81,
    )
    item = operator(
        {
            "sample_id": "omniworld/scene_a/000001_000081__ti2v",
            "scene_id": "scene_a",
            "rgb_relpath": "scene_a/color",
            "text_relpath": "scene_a/text/000001_000081.json",
            "frame_indices": list(range(81)),
            "task_type": "ti2v",
            "cond_frames": 1,
            "prompt": "",
        }
    )

    assert len(item["video"]) == 81
    assert item["video"][0].size == (832, 480)
    assert item["prompt"] == "prompt from json"
    assert item["sample_id"] == "omniworld/scene_a/000001_000081__ti2v"


def test_omniworld_training_script_defaults_to_dataset_parent():
    script_path = "examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B-OmniWorld-MistakeForcing.sh"
    with open(script_path, "r", encoding="utf-8") as file:
        script = file.read()

    match = re.search(r'DATA_ROOT="\$\{OMNIWORLD_DATA_ROOT:-(.*?)\}"', script)

    assert match is not None
    assert match.group(1) == "/mnt/workspace/hwzhang/code/dataset/OmniWorld"


def test_hidden_state_capture_builds_mean_and_selected_layers():
    capture = WanHiddenStateCapture(selected_layers=(3, 11, 19, 29))
    for layer_index in range(30):
        hidden = torch.full((1, 2, 3), float(layer_index), dtype=torch.bfloat16)
        capture.add(layer_index, hidden)

    summary = capture.finalize()

    assert summary["hidden_mean"].shape == (1, 2, 3)
    assert summary["selected_hidden_states"].shape == (4, 1, 2, 3)
    assert torch.allclose(summary["hidden_mean"].float(), torch.full((1, 2, 3), 14.5))
    assert torch.allclose(summary["selected_hidden_states"][0].float(), torch.full((1, 2, 3), 3.0))
    assert torch.allclose(summary["selected_hidden_states"][-1].float(), torch.full((1, 2, 3), 29.0))


def test_mistake_recorder_writes_tensor_and_index(tmp_path):
    recorder = WanMistakeRecorder(output_path=str(tmp_path), rank=2)
    payload = {
        "condition_embedding": torch.zeros((1, 4, 8), dtype=torch.bfloat16),
        "hidden_mean": torch.ones((1, 2, 3), dtype=torch.bfloat16),
        "selected_hidden_states": torch.ones((4, 1, 2, 3), dtype=torch.bfloat16),
        "timestep": torch.tensor([123.0], dtype=torch.bfloat16),
        "time_embedding": torch.ones((1, 6), dtype=torch.bfloat16),
        "clean_prediction": torch.ones((1, 2, 2, 2, 2), dtype=torch.bfloat16),
        "velocity_residual": torch.zeros((1, 2, 2, 2, 2), dtype=torch.bfloat16),
    }
    metadata = {
        "global_step": 7,
        "local_step": 3,
        "epoch": 1,
        "sample_id": "sample-1",
        "scene_id": "scene-a",
        "loss": 0.5,
        "timestep": 123.0,
    }

    index_record = recorder.write(payload, metadata)

    tensor_path = tmp_path / "tensors" / "rank_2" / "step_7_rank_2.pt"
    assert tensor_path.is_file()
    saved = torch.load(tensor_path, map_location="cpu", weights_only=False)
    assert saved["hidden_mean"].dtype == torch.bfloat16
    assert saved["selected_hidden_states"].shape == (4, 1, 2, 3)
    assert index_record["tensor_relpath"] == "tensors/rank_2/step_7_rank_2.pt"

    index_path = tmp_path / "index_rank_2.jsonl"
    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["sample_id"] == "sample-1"
    assert record["rank"] == 2
    assert record["shape"]["hidden_mean"] == [1, 2, 3]


class _DummyScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([10.0, 20.0], dtype=torch.float32)

    def add_noise(self, latents, noise, timestep):
        return latents + noise

    def training_target(self, latents, noise, timestep):
        return latents - noise

    def step(self, noise_pred, timestep, latents, to_final=True):
        return latents - noise_pred

    def training_weight(self, timestep):
        return torch.tensor(1.0, dtype=torch.float32)


class _DummyPipe:
    def __init__(self):
        self.scheduler = _DummyScheduler()
        self.torch_dtype = torch.bfloat16
        self.device = torch.device("cpu")
        self.in_iteration_models = ("dit",)
        self.dit = object()

    def model_fn(self, dit, input_latents=None, latents=None, context=None, timestep=None, mistake_capture=None, **kwargs):
        if mistake_capture is not None:
            mistake_capture.set_time_embedding(torch.full((1, 6), 2.0, dtype=torch.bfloat16))
            for layer_index in range(30):
                mistake_capture.add(layer_index, torch.full((1, 2, 3), float(layer_index), dtype=torch.bfloat16))
        return torch.full_like(input_latents, 0.25)


def test_flowmatch_mistake_forcing_loss_writes_expected_payload(tmp_path):
    torch.manual_seed(0)
    pipe = _DummyPipe()
    recorder = WanMistakeRecorder(output_path=str(tmp_path), rank=0)
    capture = WanHiddenStateCapture()

    loss = FlowMatchSFTMistakeForcingLoss(
        pipe,
        mistake_recorder=recorder,
        mistake_capture=capture,
        mistake_metadata={
            "global_step": 1,
            "local_step": 1,
            "epoch": 0,
            "sample_id": "sample-0",
            "scene_id": "scene-0",
        },
        input_latents=torch.ones((1, 2, 2, 2, 2), dtype=torch.bfloat16),
        context=torch.ones((1, 4, 8), dtype=torch.bfloat16),
    )

    assert loss.item() >= 0
    saved = torch.load(tmp_path / "tensors" / "rank_0" / "step_1_rank_0.pt", map_location="cpu", weights_only=False)
    assert saved["condition_embedding"].shape == (1, 4, 8)
    assert saved["hidden_mean"].shape == (1, 2, 3)
    assert saved["selected_hidden_states"].shape == (4, 1, 2, 3)
    assert saved["time_embedding"].shape == (1, 6)
    assert saved["clean_prediction"].shape == (1, 2, 2, 2, 2)
    assert saved["velocity_residual"].shape == (1, 2, 2, 2, 2)
