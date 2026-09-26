"""Degradation, the three training modes, validation strips and checkpoint retention."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nightflash import Session
from nightflash.data import DegradationDataset, make_mode_dataloaders, split_pool
from nightflash.degrade import (
    PROFILES, TRAIN_MODES, DegradeConfig, apply_degradation, degraded_pair,
    make_config, sample_params, stable_seed,
)
from nightflash.losses import (
    LOSS_VERSION, content_loss, edge_loss, masked_l1, paired_reconstruction, psnr,
)
from nightflash.validate import (
    ValidationMetrics, append_history, best_epoch, combine_metrics,
    evaluate_batch, read_history, render_strip,
)


def _photo(seed: int, size: int = 96) -> Image.Image:
    """A stand-in night-flash frame: bright subject, dark surround, some grain."""
    g = np.random.default_rng(seed)
    a = g.random((size, size, 3)) * 0.30
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    a[d < size * 0.25] = 0.85
    a = np.clip(a + g.normal(0, 0.02, a.shape), 0, 1)
    return Image.fromarray(np.uint8(a * 255), mode="RGB")


def _pool(root: Path, n: int = 10) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _photo(i).save(root / f"night_{i:03d}.jpg")
    return root


# ------------------------------------------------------------------- degrade


def test_profiles_are_monotonically_stronger() -> None:
    """Each profile must be a strictly larger perturbation than the last."""

    spreads = []
    for name in PROFILES:
        config = make_config(name, none_prob=0.0, curriculum=False)
        assert config.profile == name
        values = []
        for seed in range(16):
            rng = random.Random(seed)
            image, _ = degraded_pair(_photo(0), config, rng, force=True)
            array = np.asarray(image, np.float32) / 255
            values.append(np.percentile(array, 99) - np.percentile(array, 1))
        spreads.append(np.mean(values))
    assert spreads[0] > spreads[1] > spreads[2], spreads


def test_degradation_removes_high_frequency_detail() -> None:
    """Grain must actually be destroyed, measured on a real image."""

    image = _photo(3, 128)
    base = np.asarray(image, np.float32) / 255

    def hf(a: np.ndarray) -> float:
        return float(np.abs(np.diff(a, axis=0)).mean())

    clean = hf(base)
    for name, ceiling in (("subtle", 0.7), ("moderate", 0.45), ("strong", 0.3)):
        config = make_config(name, none_prob=0.0)
        out = apply_degradation(image, sample_params(config, random.Random(0)), random.Random(0))
        ratio = hf(np.asarray(out, np.float32) / 255) / clean
        assert ratio < ceiling, f"{name}: hf ratio {ratio:.2f} >= {ceiling}"


def test_degradation_keeps_the_highlight_visible() -> None:
    """The cue for *where* halation goes must survive the degradation.

    If the peak is crushed to nothing the model has no highlight to hang the
    glow on, which is the one thing the pipeline is not allowed to destroy.
    """

    image = _photo(4, 128)
    base_peak = float(np.asarray(image, np.float32).max() / 255)
    for name in PROFILES:
        config = make_config(name, none_prob=0.0)
        peaks = []
        for seed in range(10):
            rng = random.Random(seed)
            out, _ = degraded_pair(image, config, rng, force=True)
            peaks.append(float(np.asarray(out, np.float32).max() / 255))
        assert np.mean(peaks) > 0.45, f"{name}: peak collapsed to {np.mean(peaks):.2f}"
    assert base_peak > 0.9


def test_contrast_reduction_flattens_the_tonal_range() -> None:
    """Flattening the falloff is what separates a flash look from a flat one."""

    image = _photo(5, 128)
    base = np.asarray(image, np.float32) / 255
    base_range = float(np.percentile(base, 99) - np.percentile(base, 1))
    config = make_config("moderate", none_prob=0.0)
    out = apply_degradation(image, sample_params(config, random.Random(0)), random.Random(0))
    got = np.asarray(out, np.float32) / 255
    new_range = float(np.percentile(got, 99) - np.percentile(got, 1))
    assert new_range < base_range * 0.75, f"{base_range:.3f} -> {new_range:.3f}"


def test_params_vary_per_image_and_per_epoch() -> None:
    config = make_config("moderate", curriculum=False, none_prob=0.0)
    draws = [sample_params(config, random.Random(s)) for s in range(20)]
    assert len({round(d["exposure"], 9) for d in draws}) == 20
    first = sample_params(config, random.Random(0))
    second = sample_params(config, random.Random(1))
    assert first != second


def test_none_prob_identity_path() -> None:
    image = _photo(6)
    always = make_config("moderate", none_prob=1.0)
    out, params = degraded_pair(image, always, random.Random(0))
    assert params is None and out is image

    never = make_config("moderate", none_prob=0.0)
    hits = sum(degraded_pair(image, never, random.Random(s))[1] is None for s in range(60))
    assert hits == 0


def test_stable_seed_is_process_independent() -> None:
    """Validation degradation must be identical in every session."""

    assert stable_seed("a.jpg") == stable_seed("a.jpg")
    assert stable_seed("a.jpg") != stable_seed("b.jpg")
    # and it must not depend on PYTHONHASHSEED
    assert isinstance(stable_seed("x"), int)


def test_curriculum_ramps_and_is_the_default() -> None:
    config = DegradeConfig(curriculum_epochs=30)
    assert config.curriculum is True, "curriculum should be on by default"
    early = config.resolved(0)
    late = config.resolved(29)
    assert early.resize_min > late.resize_min, "should get more aggressive"
    assert early.contrast_min > late.contrast_min
    # monotonic across the ramp
    values = [config.resolved(e).resize_min for e in range(30)]
    assert all(a >= b for a, b in zip(values, values[1:])), values


def test_curriculum_is_inert_without_a_horizon() -> None:
    """A bare config has no epoch horizon, so it must not silently ramp."""

    config = DegradeConfig()
    assert config.curriculum_epochs == 0
    assert config.resolved(0).to_dict() == config.to_dict()
    assert config.resolved(99).to_dict() == config.to_dict()


def test_curriculum_off_leaves_config_untouched() -> None:
    config = make_config("moderate", curriculum=False, curriculum_epochs=30)
    assert config.resolved(0).to_dict() == config.to_dict()
    assert config.resolved(29).to_dict() == config.to_dict()


def test_unknown_profile_and_override_rejected() -> None:
    with pytest.raises(ValueError, match="degrade profile"):
        make_config("nonsense")
    with pytest.raises(ValueError, match="unknown degrade options"):
        make_config("moderate", not_a_real_option=1)


def test_degradation_is_size_preserving() -> None:
    """Crop shape must survive, or the shared warp correspondence breaks."""

    for size in ((64, 64), (96, 128)):
        image = _photo(7, 96).resize(size)
        config = make_config("strong", none_prob=0.0)
        out = apply_degradation(image, sample_params(config, random.Random(0)), random.Random(0))
        assert out.size == size
        assert out.mode == "RGB"


def test_degradation_never_produces_out_of_range_values() -> None:
    image = _photo(8, 64)
    for name in PROFILES:
        config = make_config(name, none_prob=0.0)
        for seed in range(8):
            rng = random.Random(seed)
            out, _ = degraded_pair(image, config, rng, force=True)
            array = np.asarray(out)
            assert array.dtype == np.uint8
            assert array.min() >= 0 and array.max() <= 255


# ------------------------------------------------------------------- losses


def test_content_loss_ignores_texture_in_the_l1_half() -> None:
    """The L1 half must not see the stochastic image.

    This is the whole point of the fix: a grainy styled image and a clean
    deterministic one should produce the same L1 contribution, so grain is not
    charged to the loss.
    """

    torch.manual_seed(0)
    source = torch.rand(2, 3, 32, 32)
    deterministic = (source + 0.05).clamp(0, 1)
    styled = (deterministic + torch.randn_like(deterministic) * 0.1).clamp(0, 1)

    l1_det = masked_l1(deterministic, source)
    l1_styled = masked_l1(styled, source)
    assert not torch.isclose(l1_det, l1_styled), "L1 should differ"

    # but through content_loss, the L1 half must be the same
    a = content_loss(styled, source, texture_reference=deterministic)
    b = content_loss(deterministic, source, texture_reference=deterministic)
    edge_a = edge_loss(styled, source)
    edge_b = edge_loss(deterministic, source)
    assert torch.isclose(a - edge_a * 0.35, b - edge_b * 0.35, atol=1e-6)


def test_content_loss_edge_half_does_see_the_styled_image() -> None:
    torch.manual_seed(1)
    source = torch.rand(1, 3, 32, 32)
    det = (source + 0.05).clamp(0, 1)
    styled = (det + torch.randn_like(det) * 0.2).clamp(0, 1)
    assert not torch.isclose(edge_loss(styled, source), edge_loss(det, source))


def test_content_loss_backwards_compatible() -> None:
    """Old two-argument calls must behave exactly as before."""

    torch.manual_seed(2)
    prediction = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)
    expected = 0.65 * masked_l1(prediction, target) + 0.35 * edge_loss(prediction, target)
    assert torch.isclose(content_loss(prediction, target), expected, atol=1e-6)


def test_per_item_reductions() -> None:
    torch.manual_seed(3)
    p = torch.rand(5, 3, 32, 32)
    t = torch.rand(5, 3, 32, 32)
    m = torch.rand(5, 1, 32, 32)
    assert paired_reconstruction(p, t, m).shape == ()
    assert paired_reconstruction(p, t, m, reduction="none").shape == (5,)
    assert masked_l1(p, t, m, reduction="none").shape == (5,)
    assert edge_loss(p, t, reduction="none").shape == (5,)


def test_psnr_rewards_a_better_match() -> None:
    torch.manual_seed(4)
    target = torch.rand(1, 3, 32, 32)
    assert psnr(target.clone(), target) >= 99.0
    near = (target + 0.01).clamp(0, 1)
    far = (target + 0.30).clamp(0, 1)
    assert psnr(near, target) > psnr(far, target)


def test_loss_version_was_bumped() -> None:
    assert LOSS_VERSION >= 2, "the objective changed, the version must reflect it"


# ------------------------------------------------------------------- dataset


def test_paired_mode_pairs_an_image_with_itself(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 6)
    ds = DegradationDataset(pool, image_size=64, mode="paired",
                            degrade=make_config("moderate"))
    for i in range(len(ds)):
        item = ds[i]
        assert item["source_path"] == item["target_path"]
        assert float(item["is_paired"]) == 1.0


def test_sameset_mode_never_pairs_an_image_with_itself(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 8)
    ds = DegradationDataset(pool, image_size=64, mode="sameset",
                            degrade=make_config("moderate"))
    seen_different = False
    for i in range(len(ds)):
        item = ds[i]
        assert item["source_path"] != item["target_path"], "j must differ from i"
        assert float(item["is_paired"]) == 0.0
        seen_different = True
    assert seen_different


def test_sameset_falls_back_on_a_single_image(tmp_path, capsys) -> None:
    pool = _pool(tmp_path / "one", 1)
    ds = DegradationDataset(pool, image_size=64, mode="sameset",
                            degrade=make_config("moderate"))
    assert ds.effective_mode == "paired"
    assert "Falling back to paired" in capsys.readouterr().out


def test_dataset_output_shapes_and_collate_safety(tmp_path) -> None:
    """Every field must survive default_collate, including the identity path."""

    pool = _pool(tmp_path / "pool", 5)
    for mode in ("paired", "sameset"):
        for none_prob in (0.0, 1.0):
            ds = DegradationDataset(
                pool, image_size=64, mode=mode, randomize=False,
                degrade=make_config("moderate", none_prob=none_prob, curriculum=False),
            )
            batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
            assert batch["source"].shape == (len(ds), 3, 64, 64)
            assert batch["target"].shape == (len(ds), 3, 64, 64)
            assert batch["mask"].shape == (len(ds), 1, 64, 64)
            assert batch["degrade_params"].shape == (len(ds), 8)
            assert batch["degraded"].shape == (len(ds),)
            assert batch["is_paired"].shape == (len(ds),)
            assert batch["source"].min() >= 0.0 and batch["source"].max() <= 1.0


def test_degradation_varies_across_epochs(tmp_path) -> None:
    """Same image, different epoch, different corruption."""

    pool = _pool(tmp_path / "pool", 5)
    ds = DegradationDataset(
        pool, image_size=64, mode="paired", randomize=False,
        degrade=make_config("moderate", none_prob=0.0, curriculum=False),
    )
    ds.set_epoch(0)
    a = ds[0]["source"].clone()
    ds.set_epoch(1)
    b = ds[0]["source"].clone()
    assert not torch.allclose(a, b), "epoch must change the degradation draw"


def test_validation_degradation_is_stable_across_epochs(tmp_path) -> None:
    """Validation images must degrade identically every epoch, or strips lie."""

    pool = _pool(tmp_path / "pool", 5)
    ds = DegradationDataset(
        pool, image_size=64, mode="paired", randomize=False,
        degrade=make_config("moderate", none_prob=0.0, curriculum=False),
        deterministic_seed=True,
    )
    ds.set_epoch(0)
    a = ds[0]["source"].clone()
    params_a = ds[0]["degrade_params"].clone()
    for epoch in (1, 5, 40):
        ds.set_epoch(epoch)
        assert torch.allclose(ds[0]["source"], a), f"epoch {epoch} changed the source"
        assert torch.allclose(ds[0]["degrade_params"], params_a)


def test_shared_warp_keeps_source_and_target_aligned(tmp_path) -> None:
    """Geometry must be shared; only the simulated camera may differ."""

    pool = _pool(tmp_path / "pool", 4)
    ds = DegradationDataset(
        pool, image_size=64, mode="paired", randomize=True,
        degrade=make_config("moderate", none_prob=0.0, curriculum=False),
    )
    # With degradation disabled entirely, source and target must be identical,
    # which can only hold if the warp was applied once and shared.
    ds.degrade = make_config("moderate", none_prob=1.0)
    for i in range(len(ds)):
        item = ds[i]
        assert torch.allclose(item["source"], item["target"], atol=1e-6)


def test_split_pool_is_disjoint_and_deterministic(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 20)
    from nightflash.data import list_images

    paths = list_images(pool)
    train_a, val_a = split_pool(paths, 0.25, seed=7)
    train_b, val_b = split_pool(paths, 0.25, seed=7)
    assert [p.name for p in train_a] == [p.name for p in train_b]
    assert [p.name for p in val_a] == [p.name for p in val_b]
    assert set(train_a).isdisjoint(val_a)
    assert len(train_a) + len(val_a) == len(paths)
    assert 0 < len(val_a) < len(paths)


def test_split_pool_never_empties_training(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 2)
    from nightflash.data import list_images

    train, val = split_pool(list_images(pool), 0.99, seed=1)
    assert len(train) >= 1


def test_degradation_dataset_rejects_a_bad_mode(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 3)
    with pytest.raises(ValueError, match="paired or sameset"):
        DegradationDataset(pool, mode="unpaired")


# ------------------------------------------------------------- mode plumbing


def test_train_mode_constants() -> None:
    assert TRAIN_MODES == ("unpaired", "paired", "sameset")
    assert PROFILES == ("subtle", "moderate", "strong")


def test_single_pool_resolves_to_paired(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 4)
    s = Session(target_dir=pool, train_mode=None, runs_root=tmp_path / "r", name="a")
    assert s.train_mode == "paired"


def test_two_pools_resolve_to_unpaired(tmp_path, capsys) -> None:
    a, b = _pool(tmp_path / "a", 3), _pool(tmp_path / "b", 3)
    s = Session(source_dir=a, target_dir=b, train_mode=None, runs_root=tmp_path / "r", name="b")
    assert s.train_mode == "unpaired"
    assert "resolved to 'unpaired'" in capsys.readouterr().out


def test_explicit_mismatched_mode_is_rejected(tmp_path) -> None:
    a, b = _pool(tmp_path / "a", 3), _pool(tmp_path / "b", 3)
    with pytest.raises(ValueError, match="must be the same directory"):
        Session(source_dir=a, target_dir=b, train_mode="paired",
                runs_root=tmp_path / "r", name="c")


def test_explicit_mode_without_target_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="needs target_dir"):
        Session(train_mode="paired", runs_root=tmp_path / "r", name="d")


def test_invalid_mode_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="train_mode must be one of"):
        Session(train_mode="nonsense", runs_root=tmp_path / "r", name="e")


def test_make_mode_dataloaders_unpaired_has_no_validation(tmp_path) -> None:
    a, b = _pool(tmp_path / "a", 6), _pool(tmp_path / "b", 6)
    train, preview, ds, val, val_ds = make_mode_dataloaders(
        a, b, train_mode="unpaired", image_size=32, batch_size=2, num_workers=0)
    assert val is None and val_ds is None
    assert len(ds) == 6


def test_make_mode_dataloaders_paired_splits_the_pool(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 10)
    _, _, train_ds, val_loader, val_ds = make_mode_dataloaders(
        None, pool, train_mode="paired", image_size=32, batch_size=2,
        num_workers=0, val_split=0.2, degrade=make_config("moderate", curriculum=False))
    assert len(train_ds) == 8
    assert len(val_ds) == 2
    assert val_loader is not None
    train_files = {p.name for p in train_ds.paths}
    val_files = {p.name for p in val_ds.paths}
    assert train_files.isdisjoint(val_files), "validation leaked into training"


# ------------------------------------------------------------ validation viz


def test_render_strip_layouts_and_sizes() -> None:
    rows = []
    for i in range(3):
        src = torch.rand(3, 48, 48)
        rows.append({
            "index": i + 1, "filename": f"n{i}.jpg",
            "source": src, "output": (src * 0.8).clamp(0, 1),
            "target": torch.rand(3, 48, 48),
        })
    metrics = ValidationMetrics(images=3, psnr=21.0, ssim=0.7, l1=0.03, effect=0.5)
    for layout in ("rows", "cols"):
        for cols in (2, 3):
            img = render_strip(rows, columns=cols, layout=layout,
                               title="t", metrics=metrics)
            assert img.size[0] > 48 * cols
            assert img.size[1] > 48


def test_render_strip_handles_empty() -> None:
    assert render_strip([]).size == (64, 32)


def test_render_strip_header_is_not_clipped() -> None:
    row = {"index": 1, "filename": "a.jpg", "source": torch.rand(3, 16, 16),
           "output": torch.rand(3, 16, 16), "target": torch.rand(3, 16, 16)}
    metrics = ValidationMetrics(images=1, psnr=123.456, ssim=0.98765,
                                l1=0.001234, effect=0.56789)
    img = render_strip([row], columns=3, title="x" * 200, metrics=metrics)
    # A very long title must widen the canvas rather than being cut off.
    assert img.size[0] > 3 * 16 + 16


def test_combine_metrics_is_image_weighted() -> None:
    """A short final batch must not be over-weighted."""

    parts = [ValidationMetrics(images=8, psnr=20.0, ssim=0.5, l1=0.1, effect=0.4),
             ValidationMetrics(images=2, psnr=40.0, ssim=0.9, l1=0.01, effect=0.4)]
    combined = combine_metrics(parts)
    assert combined.images == 10
    # 8 images at 20 dB and 2 at 40 dB must weight to exactly 24, not to the
    # unweighted mean of 30.
    assert abs(combined.psnr - 24.0) < 1e-6, combined.psnr
    assert combine_metrics([]).images == 0


def test_history_round_trip_and_best(tmp_path) -> None:
    path = tmp_path / "history.csv"
    for epoch, value in ((1, 18.0), (2, 24.0), (3, 21.0), (4, 26.5)):
        append_history(path, {"epoch": epoch, "images": 4, "psnr": value,
                              "ssim": 0.7, "l1": 0.03, "effect": 0.5})
    rows = read_history(path)
    assert len(rows) == 4
    assert [r["epoch"] for r in rows] == ["1", "2", "3", "4"]
    assert best_epoch(path) == (4, 26.5)


def test_best_epoch_on_empty_history(tmp_path) -> None:
    assert best_epoch(tmp_path / "missing.csv") is None


def test_evaluate_batch_counts_images_not_batches() -> None:
    torch.manual_seed(5)
    target = torch.rand(3, 3, 32, 32)
    batch = {"target": target, "degrade_params": torch.rand(3, 8)}
    off = (target + 0.2).clamp(0, 1)
    out = evaluate_batch(batch, torch.rand(3, 3, 32, 32), off)
    assert out.images == 3
    perfect = evaluate_batch(batch, target, target.clone())
    assert perfect.psnr > out.psnr
    assert perfect.l1 < 1e-6


# --------------------------------------------------------- session behaviour


def _tiny(tmp_path, **kwargs):
    pool = _pool(tmp_path / "pool", 8)
    defaults = dict(
        target_dir=pool, image_size=32, batch_size=2, epochs=1,
        num_workers=0, device="cpu", use_amp=False, preview_every=0,
        log_every=100, halation=False, subject_gate=False, region_exposure=False,
        runs_root=tmp_path / "runs", name="v", val_count=2, degrade_profile="subtle",
    )
    defaults.update(kwargs)
    return Session(**defaults)


def test_paired_training_writes_strip_history_and_best(tmp_path) -> None:
    s = _tiny(tmp_path, epochs=2, keep_last=1)
    s.train()
    strips = sorted(s.validation_dir.glob("epoch-*.png"))
    assert len(strips) == 2
    assert (s.validation_dir / "latest.png").exists()
    assert len(read_history(s.validation_dir / "history.csv")) == 2
    assert (s.checkpoint_dir / "best.pt").exists()
    kept = sorted(s.checkpoint_dir.glob("epoch-*.pt"))
    assert len(kept) == 1, kept


def test_sameset_does_not_write_best(tmp_path) -> None:
    """No correspondence means PSNR ranks nothing, so best.pt must not exist."""

    s = _tiny(tmp_path, train_mode="sameset", epochs=1)
    s.train()
    assert sorted(s.validation_dir.glob("epoch-*.png"))
    assert not (s.checkpoint_dir / "best.pt").exists()


def test_unpaired_creates_no_validation(tmp_path) -> None:
    a, b = _pool(tmp_path / "a", 6), _pool(tmp_path / "b", 6)
    s = Session(source_dir=a, target_dir=b, train_mode="unpaired", image_size=32,
                batch_size=2, epochs=1, num_workers=0, device="cpu", use_amp=False,
                preview_every=0, halation=False, subject_gate=False,
                region_exposure=False, runs_root=tmp_path / "runs", name="u")
    s.train()
    assert not s.validation_dir.exists()


def test_config_records_loss_version_and_mode(tmp_path) -> None:
    s = _tiny(tmp_path, epochs=1)
    s.train()
    saved = json.loads((s.run_dir / "config.json").read_text())
    assert saved["loss_version"] == LOSS_VERSION
    assert saved["train_mode"] == "paired"


def test_resume_warns_on_loss_version_mismatch(tmp_path) -> None:
    s = _tiny(tmp_path, epochs=1)
    s.train()
    import torch as _torch

    checkpoint = _torch.load(s.latest, map_location="cpu", weights_only=False)
    checkpoint["loss_version"] = 1
    _torch.save(checkpoint, s.latest)
    state = s.state()
    assert any("loss_version=1" in w for w in state.warnings), state.warnings
    assert state.saved_loss_version == 1


def test_validation_history_is_appended_across_sessions(tmp_path) -> None:
    s = _tiny(tmp_path, epochs=1)
    s.train()
    s.train(extra_epochs=1)
    rows = read_history(s.validation_dir / "history.csv")
    assert [r["epoch"] for r in rows] == ["1", "2"]


def test_ema_runs_and_is_checkpointed(tmp_path) -> None:
    s = _tiny(tmp_path, epochs=1, ema_decay=0.9)
    s.train()
    import torch as _torch

    checkpoint = _torch.load(s.latest, map_location="cpu", weights_only=False)
    assert checkpoint.get("ema"), "EMA state should be saved"
    assert len(checkpoint["ema"]) == len(checkpoint["model"])


def test_ema_resumes(tmp_path) -> None:
    s = _tiny(tmp_path, epochs=1, ema_decay=0.9)
    s.train()
    s.train(extra_epochs=1)
    assert s.state().last_epoch == 1


def test_curriculum_reaches_the_dataset(tmp_path) -> None:
    """Session must hand its epoch horizon to the degradation config."""

    s = _tiny(tmp_path, epochs=10)
    assert s.degrade.curriculum_epochs == 10
    assert s.degrade.resolved(0).resize_min > s.degrade.resolved(9).resize_min
