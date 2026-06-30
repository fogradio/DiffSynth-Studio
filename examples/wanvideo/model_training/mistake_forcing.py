import json
import os

import torch

from diffsynth.core.data.operators import LoadImage


def _load_prompt_from_text_json(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    captions = payload.get("captions", {})
    for key in ("Video_Caption", "video_caption", "caption", "prompt"):
        value = captions.get(key, payload.get(key))
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Cannot resolve prompt from text manifest: {path}")


class OmniWorldFrameSequenceOperator:
    def __init__(self, base_path, frame_processor, num_frames=81):
        self.base_path = base_path
        self.frame_processor = frame_processor
        self.num_frames = num_frames
        self.image_loader = LoadImage()

    def _frame_path(self, rgb_relpath, frame_index):
        return os.path.join(self.base_path, rgb_relpath, f"{int(frame_index):06d}.png")

    def __call__(self, sample):
        sample = sample.copy()
        task_type = sample.get("task_type")
        cond_frames = sample.get("cond_frames")
        if task_type is not None and task_type != "ti2v":
            raise ValueError(f"Unsupported OmniWorld task_type for Wan T2V SFT: {task_type}")
        if cond_frames is not None and int(cond_frames) != 1:
            raise ValueError(f"Unsupported OmniWorld cond_frames for Wan T2V SFT: {cond_frames}")
        frame_indices = list(sample.get("frame_indices", []))
        if len(frame_indices) < self.num_frames:
            raise ValueError(f"Expected at least {self.num_frames} frame indices, got {len(frame_indices)}")
        frame_indices = frame_indices[: self.num_frames]
        frames = []
        for frame_index in frame_indices:
            frame_path = self._frame_path(sample["rgb_relpath"], frame_index)
            frame = self.image_loader(frame_path)
            frame = self.frame_processor(frame)
            frames.append(frame)
        prompt = sample.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = _load_prompt_from_text_json(os.path.join(self.base_path, sample["text_relpath"]))
        sample["frame_indices"] = frame_indices
        sample["video"] = frames
        sample["prompt"] = prompt
        return sample


class OmniWorldManifestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_operator):
        self.dataset = dataset
        self.sample_operator = sample_operator
        self.load_from_cache = dataset.load_from_cache

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.sample_operator(self.dataset[index])


class WanHiddenStateCapture:
    def __init__(self, selected_layers=(3, 11, 19, 29), detach=True):
        self.selected_layers = tuple(selected_layers)
        self.hidden_sum = None
        self.hidden_count = 0
        self.selected_hidden_states = {}
        self.time_embedding = None
        self.hidden_dtype = None
        # When False, captured hidden states / time embedding keep their grad_fn
        # so an auxiliary head (copilot) can backprop into the DiT backbone.
        # Default True preserves the original detached-dump behaviour.
        self.detach = detach

    def set_time_embedding(self, time_embedding):
        self.time_embedding = time_embedding.detach() if self.detach else time_embedding

    def add(self, layer_index, hidden_states):
        if self.detach:
            hidden_states = hidden_states.detach()
        if self.hidden_dtype is None:
            self.hidden_dtype = hidden_states.dtype
        if self.hidden_sum is None:
            self.hidden_sum = torch.zeros_like(hidden_states, dtype=torch.float32)
        self.hidden_sum = self.hidden_sum + hidden_states.float()
        self.hidden_count += 1
        if layer_index in self.selected_layers:
            self.selected_hidden_states[layer_index] = hidden_states.clone()

    def finalize(self):
        if self.hidden_sum is None or self.hidden_count == 0:
            raise ValueError("No hidden states were captured.")
        missing_layers = [layer for layer in self.selected_layers if layer not in self.selected_hidden_states]
        if missing_layers:
            raise ValueError(f"Missing selected hidden states for layers: {missing_layers}")
        return {
            "hidden_mean": (self.hidden_sum / self.hidden_count).to(self.hidden_dtype),
            "selected_hidden_states": torch.stack(
                [self.selected_hidden_states[layer] for layer in self.selected_layers],
                dim=0,
            ),
            "time_embedding": self.time_embedding,
        }


class WanMistakeRecorder:
    def __init__(self, output_path, rank):
        self.output_path = output_path
        self.rank = int(rank)
        self.tensor_dir = os.path.join(self.output_path, "tensors", f"rank_{self.rank}")
        self.index_path = os.path.join(self.output_path, f"index_rank_{self.rank}.jsonl")
        os.makedirs(self.tensor_dir, exist_ok=True)

    def _to_cpu_payload(self, payload):
        payload_cpu = {}
        for key, value in payload.items():
            if torch.is_tensor(value):
                payload_cpu[key] = value.detach().to("cpu")
            else:
                payload_cpu[key] = value
        return payload_cpu

    def write(self, payload, metadata):
        payload = self._to_cpu_payload(payload)
        global_step = int(metadata["global_step"])
        tensor_file_name = f"step_{global_step}_rank_{self.rank}.pt"
        tensor_relpath = os.path.join("tensors", f"rank_{self.rank}", tensor_file_name)
        tensor_path = os.path.join(self.output_path, tensor_relpath)
        torch.save(payload, tensor_path)

        record = {
            "tensor_relpath": tensor_relpath,
            "rank": self.rank,
            "local_step": int(metadata["local_step"]),
            "global_step": global_step,
            "epoch": int(metadata["epoch"]),
            "sample_id": metadata.get("sample_id"),
            "scene_id": metadata.get("scene_id"),
            "timestep": float(metadata["timestep"]),
            "loss": float(metadata["loss"]),
            "shape": {
                key: list(value.shape)
                for key, value in payload.items()
                if torch.is_tensor(value)
            },
            "dtype": {
                key: str(value.dtype).replace("torch.", "")
                for key, value in payload.items()
                if torch.is_tensor(value)
            },
        }
        with open(self.index_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
        return record
