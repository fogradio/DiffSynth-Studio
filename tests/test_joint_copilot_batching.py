import ast
from pathlib import Path

import torch

from examples.wanvideo.model_training.train_joint_copilot import (
    InMemoryCopilotTrainer,
)
from examples.wanvideo.model_training.wisa_dataset import WISAVideoOperator


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "wanvideo"
    / "model_training"
    / "train_joint_copilot.py"
)


def _training_tree():
    return ast.parse(TRAINING_SCRIPT.read_text(encoding="utf-8"))


def _is_attr(node, value, attr):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == value
        and node.attr == attr
    )


def test_joint_dataloader_uses_requested_batch_size_and_preserves_samples():
    tree = _training_tree()
    dataloader_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "DataLoader"
    ]

    assert len(dataloader_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in dataloader_calls[0].keywords}
    assert ast.unparse(keywords["batch_size"]) == "args.batch_size"
    collate = keywords["collate_fn"]
    assert isinstance(collate, ast.Lambda)
    assert isinstance(collate.body, ast.Name)
    assert collate.body.id == collate.args.args[0].arg


def test_cached_bs1_batch_is_unwrapped_before_model_forward():
    tree = _training_tree()
    cached_model_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _is_attr(
            node.test, "dataset", "load_from_cache"
        ):
            continue
        cached_model_calls.extend(
            child
            for statement in node.body
            for child in ast.walk(statement)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "model"
        )

    assert len(cached_model_calls) == 1
    inputs = {
        keyword.arg: keyword.value for keyword in cached_model_calls[0].keywords
    }["inputs"]
    assert isinstance(inputs, ast.Subscript)
    assert isinstance(inputs.value, ast.Name) and inputs.value.id == "data"
    assert isinstance(inputs.slice, ast.Constant) and inputs.slice.value == 0


def test_wisa_batching_enables_fixed_frame_sampling():
    tree = _training_tree()
    operator_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "WISAVideoOperator"
    ]

    assert len(operator_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in operator_calls[0].keywords}
    assert ast.unparse(keywords["fixed_frames"]) == "args.batch_size > 1"

    operator = WISAVideoOperator(
        name2path={}, frame_processor=None, max_frames=9, fixed_frames=True
    )
    indices = operator._sample_indices(total_frames=3)
    assert len(indices) == 9
    assert min(indices) == 0
    assert max(indices) == 2


def test_joint_training_script_has_no_temporary_debug_markers():
    source = TRAINING_SCRIPT.read_text(encoding="utf-8")
    assert "DBG-" not in source
    assert "DEBUG-" not in source


def test_v3_selected_hidden_states_keep_batch_and_layer_axes_distinct():
    class RecordingCopilot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.selected_hidden_states = None

        def forward(
            self,
            clean_prediction,
            time_embedding,
            condition_embedding,
            selected_hidden_states,
        ):
            self.selected_hidden_states = selected_hidden_states.clone()
            return torch.zeros_like(clean_prediction)

    copilot = RecordingCopilot()
    trainer = InMemoryCopilotTrainer(copilot, copilot_version="v3")
    layer_first = torch.arange(4 * 4 * 2).reshape(4, 4, 2, 1).float()
    payload = {
        "clean_prediction": torch.zeros(4, 1, 1, 1, 1),
        "velocity_residual": torch.zeros(4, 1, 1, 1, 1),
        "time_embedding": torch.zeros(4, 6),
        "condition_embedding": torch.zeros(4, 2, 8),
        "selected_hidden_states": layer_first,
    }

    trainer.write(payload, metadata={})

    expected_batch_first = layer_first.permute(1, 0, 2, 3).contiguous()
    assert torch.equal(copilot.selected_hidden_states, expected_batch_first)
