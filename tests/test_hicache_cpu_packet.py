"""CPU-only correctness packet for the HiCache++ sampler adapter."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_SOURCE = Path(__file__).parents[1] / "trellis" / "pipelines" / "samplers" / "hicache.py"
_SPEC = importlib.util.spec_from_file_location("hicache_cpu_packet", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
hicache = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hicache)


class CountingModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, t, cond=None, **kwargs):
        self.calls += 1
        return torch.full_like(x, float(self.calls))


class FakeSampler:
    sigma_min = 0.1

    def sample(self, model, noise, cond=None, steps=8, **kwargs):
        x = noise
        for step in range(steps):
            out = self.sample_once(model, x, 1.0 - step / steps, 1.0 - (step + 1) / steps, cond)
            x = out["pred_x_prev"]
        return SimpleNamespace(samples=x)

    def sample_once(self, model, x_t, t, t_prev, cond=None, **kwargs):
        pred_x_0, _eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        return {"pred_x_prev": x_t - (t - t_prev) * pred_v, "pred_x_0": pred_x_0}

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = model(x_t, t, cond, **kwargs)
        return x_t - pred_v, None, pred_v

    def _v_to_xstart_eps(self, x_t, t, v):
        return x_t - v, None


def _install(pipeline, **kwargs):
    # The adapter's only EasyDict use is its two-field return container.
    easydict = types.ModuleType("easydict")
    easydict.EasyDict = dict
    sys.modules["easydict"] = easydict
    return hicache.enable(pipeline, patch_slat=False, **kwargs)


def _assert_report_labels(manifest):
    assert len(manifest["events"]) == len(manifest["measurements"])
    for event, measurement in zip(manifest["events"], manifest["measurements"]):
        assert event["stage"] == measurement["stage"]
        assert event["method"] == measurement["method"]
    scope = manifest["adapter_measurement_scope"]
    assert "input tensor bytes" in scope["memory_mb"]
    assert "error-capped budgets are rejected" in scope["predicted_error"]


def test_selected_dmd_warmup_and_manifest_labels_match_execution():
    sampler = FakeSampler()
    pipeline = SimpleNamespace(sparse_structure_sampler=sampler)
    _install(pipeline, backend="dmd", interval=2, first_enhance=1)

    sampler.sample(CountingModel(), torch.zeros(1, 2), steps=8)
    manifest = sampler.get_hicache_manifest()

    assert sampler._hicache is None
    assert sampler._budget_runtime is None
    assert manifest["budget"]["backend"] == "dmd"
    assert "hermite" in [row["method"] for row in manifest["measurements"]]
    assert "dmd" in [row["method"] for row in manifest["measurements"]]
    assert manifest["counts"]["forecast"] > 0
    _assert_report_labels(manifest)


def test_budget_guard_forces_full_and_cleans_up():
    sampler = FakeSampler()
    pipeline = SimpleNamespace(sparse_structure_sampler=sampler)
    _install(
        pipeline,
        backend="dmd",
        interval=3,
        first_enhance=1,
        budget={
            "backend": "dmd",
            "allowed_stages": ("sparse_structure",),
            "max_horizon": 1,
            "fallback": "full",
        },
    )

    sampler.sample(CountingModel(), torch.zeros(1, 2), steps=5)
    manifest = sampler.get_hicache_manifest()

    assert sampler._hicache is None
    assert sampler._budget_runtime is None
    assert manifest["counts"]["fallback"] > 0
    fallback_index = next(i for i, event in enumerate(manifest["events"]) if event["mode"] == "fallback")
    fallback = manifest["events"][fallback_index]
    assert fallback["reason"] == "horizon_exceeds_budget"
    assert fallback["method"] == "full"
    assert manifest["measurements"][fallback_index]["method"] == "full"
    _assert_report_labels(manifest)


@pytest.mark.parametrize(
    "budget, message",
    [
        ({"backend": "hermite"}, "does not match"),
        ({"backend": "auto"}, "does not match"),
        ({"backend": "dmd", "fallback": "hermite"}, "fallback='full'"),
        ({"backend": "dmd", "max_predicted_error": 0.1}, "max_predicted_error"),
    ],
)
def test_rejects_unimplemented_budget_mapping_at_install_and_reconfigure(budget, message):
    sampler = FakeSampler()
    pipeline = SimpleNamespace(sparse_structure_sampler=sampler)
    with pytest.raises(ValueError, match=message):
        _install(pipeline, backend="dmd", budget=budget)

    sampler = FakeSampler()
    pipeline = SimpleNamespace(sparse_structure_sampler=sampler)
    _install(pipeline, backend="dmd")
    with pytest.raises(ValueError, match=message):
        sampler.configure_hicache_budget(budget=budget)
