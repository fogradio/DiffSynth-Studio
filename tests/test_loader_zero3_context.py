import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch


def load_loader_model_module(monkeypatch):
    @contextmanager
    def fake_skip_model_initialization():
        yield

    package_names = [
        "diffsynth",
        "diffsynth.core",
        "diffsynth.core.loader",
        "diffsynth.core.vram",
    ]
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    fake_initialization = types.ModuleType("diffsynth.core.vram.initialization")
    fake_initialization.skip_model_initialization = fake_skip_model_initialization
    monkeypatch.setitem(sys.modules, "diffsynth.core.vram.initialization", fake_initialization)

    fake_disk_map = types.ModuleType("diffsynth.core.vram.disk_map")
    fake_disk_map.DiskMap = object
    monkeypatch.setitem(sys.modules, "diffsynth.core.vram.disk_map", fake_disk_map)

    fake_layers = types.ModuleType("diffsynth.core.vram.layers")
    fake_layers.enable_vram_management = lambda model, *args, **kwargs: model
    monkeypatch.setitem(sys.modules, "diffsynth.core.vram.layers", fake_layers)

    fake_file = types.ModuleType("diffsynth.core.loader.file")
    fake_file.load_state_dict = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "diffsynth.core.loader.file", fake_file)

    module_path = Path(__file__).resolve().parents[1] / "diffsynth" / "core" / "loader" / "model.py"
    spec = importlib.util.spec_from_file_location("diffsynth.core.loader.model", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_zero3_init_context_uses_transformers_deepspeed_config(monkeypatch):
    captured = {}

    class FakeZeroInit:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    fake_deepspeed = types.SimpleNamespace(zero=types.SimpleNamespace(Init=FakeZeroInit))
    deepspeed_config = {"zero_optimization": {"stage": 3}}

    import transformers.integrations.deepspeed as transformers_deepspeed

    loader_model = load_loader_model_module(monkeypatch)
    monkeypatch.setattr(loader_model, "is_deepspeed_zero3_enabled", lambda: True)
    monkeypatch.setattr(transformers_deepspeed, "deepspeed_config", lambda: deepspeed_config)
    monkeypatch.setitem(sys.modules, "deepspeed", fake_deepspeed)

    init_contexts = loader_model.get_init_context(torch_dtype=torch.bfloat16, device="cuda")

    assert len(init_contexts) == 1
    with init_contexts[0]:
        pass
    assert captured["kwargs"] == {"config_dict_or_path": deepspeed_config}


def test_zero3_load_model_uses_transformers_modeling_utils_loader(monkeypatch):
    loader_model = load_loader_model_module(monkeypatch)
    captured = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def to(self, *args, **kwargs):
            captured["to"] = {"args": args, "kwargs": kwargs}
            return self

    def fake_load_state_dict_into_model(model, state_dict, start_prefix, assign_to_params_buffers=False):
        captured["loader"] = {
            "model": model,
            "state_dict": state_dict,
            "start_prefix": start_prefix,
            "assign_to_params_buffers": assign_to_params_buffers,
        }
        return []

    monkeypatch.setattr(loader_model, "is_deepspeed_zero3_enabled", lambda: True)
    monkeypatch.setattr(loader_model, "get_init_context", lambda torch_dtype, device: [])
    monkeypatch.delattr("transformers.integrations.deepspeed._load_state_dict_into_zero3_model", raising=False)
    fake_modeling_utils = types.ModuleType("transformers.modeling_utils")
    fake_modeling_utils._load_state_dict_into_model = fake_load_state_dict_into_model
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_modeling_utils)

    state_dict = {"weight": torch.ones(1)}
    model = loader_model.load_model(
        FakeModel,
        path="unused.safetensors",
        torch_dtype=torch.bfloat16,
        device="cuda",
        state_dict=state_dict,
    )

    assert captured["loader"]["model"] is model
    assert captured["loader"]["state_dict"] == state_dict
    assert captured["loader"]["start_prefix"] == ""
    assert captured["loader"]["assign_to_params_buffers"] is False
    assert captured["to"]["kwargs"] == {"dtype": torch.bfloat16, "device": "cuda"}
