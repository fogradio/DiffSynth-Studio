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
