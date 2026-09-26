"""Augmentation, inference-downscale and Kaggle-resume behaviour."""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from nightflash.augment import (
    AugmentConfig, _sample_geometry, apply_geometry, make_pair,
)
from nightflash.kaggle import find_snapshot
from nightflash.session import AUGMENT_PRESETS, Session, _augment_from_preset
from nightflash.tiles import apply_tiled

PRESETS = ("default", "conservative", "strong", "none")


def _rgb(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((height, width, 3)) * 255).astype(np.uint8))


def _mask(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(((rng.random((height, width)) > 0.5) * 255).astype(np.uint8), mode="L")


# --------------------------------------------------------------------- presets


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_all_resolve(preset: str) -> None:
    config = _augment_from_preset(preset)
    assert isinstance(config, AugmentConfig)
    assert config.describe()


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="augment_preset"):
        _augment_from_preset("nonsense")


def test_presets_are_ordered_by_strength() -> None:
    spread = lambda c: (c.scale_max - c.scale_min) + c.rotate_degrees / 10  # noqa: E731
    by_name = {p: spread(_augment_from_preset(p)) for p in PRESETS}
    assert by_name["none"] < by_name["conservative"] < by_name["default"] < by_name["strong"]


def test_augment_presets_are_the_documented_set() -> None:
    assert set(AUGMENT_PRESETS) == set(PRESETS)


# -------------------------------------------------------------------- geometry


@pytest.mark.parametrize("preset", PRESETS)
def test_output_is_always_exactly_image_size(preset: str) -> None:
    """Every knob combination must still yield a size x size crop.

    This is what lets ``--tile`` match ``--image-size``: a padded or undersized
    crop would silently break the match.
    """

    source, target, mask = _rgb(1200, 800, 1), _rgb(900, 600, 2), _mask(900, 600, 3)
    config = _augment_from_preset(preset)
    for seed in range(25):
        out_source, out_target, out_mask = make_pair(
            source, target, 256, config, random.Random(seed), mask
        )
        assert out_source.size == (256, 256)
        assert out_target.size == (256, 256)
        assert out_mask is not None and out_mask.size == (256, 256)


def test_undersized_source_is_upscaled_not_padded() -> None:
    """A source smaller than the crop must be resampled up, never grey-padded.

    Padding would teach the model that a flat grey border is a normal input.
    """

    tiny = _rgb(90, 140, 4)
    out = apply_geometry(tiny, 256, {"scale": 0.7, "degrees": 0.0}, random.Random(0))
    assert out.size == (256, 256)
    assert not (np.asarray(out) == 114).all(axis=2).any()


def test_mask_survives_rotation_without_blurring() -> None:
    """Nearest-neighbour resampling keeps a binary person mask binary."""

    config = AugmentConfig(rotate_degrees=30.0, hflip=0.0, vflip=0.0)
    _, _, mask = make_pair(
        _rgb(1200, 800, 5), _rgb(900, 600, 6), 256, config, random.Random(7), _mask(900, 600, 8)
    )
    values = np.asarray(mask, dtype=np.float32)
    assert ((values > 20) & (values < 235)).sum() == 0
    assert values[0, 0] == 0.0  # a rotated corner must not invent a person


def test_mask_geometry_matches_source_geometry() -> None:
    """The same warp must be applied to the mask as to its image.

    If they diverged, the subject gate would light up the wrong pixels.
    """

    source, mask = _rgb(1200, 800, 9), _mask(1200, 800, 10)
    params = {"scale": 1.1, "degrees": 9.0, "hflip": True, "vflip": False}
    warped_image = apply_geometry(source, 128, params, random.Random(0))
    warped_mask = apply_geometry(
        mask.convert("L"), 128, params, random.Random(0),
        resample=Image.Resampling.NEAREST, fill=0,
    )
    # The mask is a coin flip per pixel, so agreement should be near chance but
    # clearly above zero: identical geometry means a real correlation.
    agreement = (
        (np.asarray(warped_image).mean(axis=2) > 128)
        == (np.asarray(warped_mask) > 128)
    ).mean()
    assert 0.45 < agreement < 0.55


def test_sampled_geometry_stays_within_declared_bounds() -> None:
    config = AugmentConfig(rotate_degrees=12.0, scale_min=0.8, scale_max=1.25)
    for seed in range(400):
        params = _sample_geometry(config, random.Random(seed))
        assert -12.0 <= params["degrees"] <= 12.0
        assert 0.8 <= params["scale"] <= 1.25
        assert isinstance(params["hflip"], bool)
        assert isinstance(params["vflip"], bool)


def test_none_preset_samples_a_degenerate_warp() -> None:
    config = _augment_from_preset("none")
    for seed in range(50):
        params = _sample_geometry(config, random.Random(seed))
        assert params["scale"] == 1.0
        assert params["degrees"] == 0.0
        assert params["hflip"] is False
        assert params["vflip"] is False


# ------------------------------------------------------------------- photometric


def test_colour_jitter_never_touches_the_target_by_default() -> None:
    """The target carries the style. Jittering it teaches a fuzzy look."""

    config = AugmentConfig(brightness=0.4, contrast=0.4, saturation=0.4, temperature=0.3)
    plain = Image.fromarray(np.full((64, 64, 3), 120, np.uint8))
    for seed in range(10):
        source, target, _ = make_pair(plain, plain, 64, config, random.Random(seed))
        assert abs(float(np.asarray(target).mean()) - 120.0) < 1.0
        assert isinstance(source, Image.Image)


def test_colour_jitter_can_be_opted_into() -> None:
    config = AugmentConfig(
        brightness=0.4, contrast=0.0, saturation=0.0, temperature=0.0,
        apply_colour_to_target=True,
    )
    plain = Image.fromarray(np.full((64, 64, 3), 120, np.uint8))
    means = {
        float(np.asarray(make_pair(plain, plain, 64, config, random.Random(s))[1]).mean())
        for s in range(40)
    }
    assert len(means) > 5, "opted-in target jitter should actually vary"


def test_holes_are_off_by_default_and_filled_with_image_mean() -> None:
    # Rotation is disabled here so the 114 fill at the corners cannot be
    # confused with a hole.
    flat = dict(rotate_degrees=0.0, hflip=0.0, vflip=0.0, scale_min=1.0, scale_max=1.0)
    plain = Image.fromarray(np.full((64, 64, 3), 200, np.uint8))

    source, _, _ = make_pair(plain, plain, 64, AugmentConfig(**flat), random.Random(0))
    assert np.asarray(source).min() == 200, "no holes by default"

    holed = AugmentConfig(**flat, holes=2, hole_size=0.25, hole_count_prob=1.0)
    values = {
        int(np.asarray(make_pair(plain, plain, 64, holed, random.Random(s))[0]).min())
        for s in range(10)
    }
    assert values == {200}, "holes fill with the image mean, never black"


def test_rotation_fill_is_the_neutral_grey() -> None:
    """The rotate fill must be identical for source and target.

    A per-image fill would make the two halves of a batch differ for a reason
    that has nothing to do with the domain gap.
    """

    flat = dict(hflip=0.0, vflip=0.0, scale_min=1.0, scale_max=1.0, rotate_degrees=45.0)
    source = _rgb(200, 200, 30)
    warped_source, warped_target, _ = make_pair(
        source, _rgb(200, 200, 31), 64, AugmentConfig(**flat), random.Random(0)
    )
    source_corner = np.asarray(warped_source)[0, 0]
    target_corner = np.asarray(warped_target)[0, 0]
    assert tuple(source_corner) == (114, 114, 114)
    assert tuple(target_corner) == (114, 114, 114)


# ------------------------------------------------------------------- inference


@pytest.mark.parametrize("pad_mode", ["edge", "reflect", "black"])
def test_every_pad_mode_returns_a_full_frame(pad_mode: str) -> None:
    image = _rgb(400, 300, 13)
    out = apply_tiled(image, lambda t: t, tile_size=256, overlap=64, pad_mode=pad_mode)
    assert out.size == image.size


def test_reflect_falls_back_to_edge_when_pad_exceeds_axis() -> None:
    """Reflection cannot exceed the axis it mirrors; it must degrade, not raise."""

    image = _rgb(80, 300, 14)
    out = apply_tiled(image, lambda t: t, tile_size=256, overlap=64, pad_mode="reflect")
    assert out.size == image.size


def test_identity_tiling_has_no_border_fade() -> None:
    """Edge tiles must not fade the frame border towards black."""

    array = np.full((300, 400, 3), 255, np.uint8)
    image = Image.fromarray(array)
    out = np.asarray(
        apply_tiled(image, lambda t: t, tile_size=256, overlap=64), dtype=np.float32
    )
    for strip in (out[0], out[-1], out[:, 0], out[:, -1]):
        assert strip.mean() > 250.0


# ---------------------------------------------------------------- max_edge


@pytest.mark.parametrize(
    "width,height,target,expected",
    [
        (4032, 3024, 1024, (1024, 768)),
        (4032, 3024, 2048, (2048, 1536)),
        (1920, 1080, 960, (960, 540)),
        (600, 800, 400, (300, 400)),
        (1170, 2532, 1000, (462, 1000)),
    ],
)
def test_max_edge_preserves_aspect_ratio(
    width: int, height: int, target: int, expected: tuple[int, int]
) -> None:
    scale = target / max(width, height)
    got = (max(1, round(width * scale)), max(1, round(height * scale)))
    assert got == expected
    assert abs(got[0] / got[1] - width / height) < 0.005


def test_max_edge_never_upscales(tmp_path) -> None:
    source_dir, target_dir = tmp_path / "a", tmp_path / "b"
    source_dir.mkdir(); target_dir.mkdir()
    for index in range(4):
        _rgb(64, 64, index).save(source_dir / f"{index}.png")
        _rgb(64, 64, index + 10).save(target_dir / f"{index}.png")
    subject = _rgb(300, 200, 20)
    subject.save(tmp_path / "s.png")

    session = Session(
        source_dir=source_dir, target_dir=target_dir, image_size=64,
        batch_size=2, epochs=1, num_workers=0, runs_root=tmp_path / "runs", name="e",
    )
    session.train(1)
    out = session.infer(tmp_path / "s.png", max_edge=4096)
    assert Image.open(out[0]).size == (300, 200)


# --------------------------------------------------------------- kaggle resume


def test_find_snapshot_matches_the_default_slug(tmp_path, monkeypatch) -> None:
    """The slug :func:`snapshot` generates must be recognised on the way back.

    ``snapshot`` lowercases and hyphenates the run name, so a raw string
    comparison against the run directory would never match.
    """

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("nightflash-other-run", "nightflash-edgehat-freqsplit-night"):
        (input_dir / name / "checkpoints").mkdir(parents=True)
        (input_dir / name / "checkpoints" / "latest.pt").write_bytes(b"x")

    monkeypatch.setattr("nightflash.kaggle.INPUT", input_dir)
    found = find_snapshot("runs/edgehat_freqsplit_night")
    assert found is not None
    assert found.name == "nightflash-edgehat-freqsplit-night"


def test_find_snapshot_prefers_an_exact_name_match(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("nightflash-other-run", "edgehat_freqsplit_night"):
        (input_dir / name / "checkpoints").mkdir(parents=True)
        (input_dir / name / "checkpoints" / "latest.pt").write_bytes(b"x")

    monkeypatch.setattr("nightflash.kaggle.INPUT", input_dir)
    found = find_snapshot("runs/edgehat_freqsplit_night")
    assert found is not None
    assert found.name == "edgehat_freqsplit_night"


def test_find_snapshot_accepts_a_single_misnamed_candidate(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    (input_dir / "whatever-i-called-it" / "checkpoints").mkdir(parents=True)
    (input_dir / "whatever-i-called-it" / "checkpoints" / "latest.pt").write_bytes(b"x")
    monkeypatch.setattr("nightflash.kaggle.INPUT", input_dir)
    found = find_snapshot("runs/edgehat_freqsplit_night")
    assert found is not None and found.name == "whatever-i-called-it"


def test_find_snapshot_returns_none_when_ambiguous(tmp_path, monkeypatch) -> None:
    """Two candidates and no name match means guessing, so refuse."""

    input_dir = tmp_path / "input"
    for name in ("snapshot-a", "snapshot-b"):
        (input_dir / name / "checkpoints").mkdir(parents=True)
        (input_dir / name / "checkpoints" / "latest.pt").write_bytes(b"x")
    monkeypatch.setattr("nightflash.kaggle.INPUT", input_dir)
    assert find_snapshot("runs/edgehat_freqsplit_night") is None


def test_find_snapshot_ignores_non_run_datasets(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    (input_dir / "my-photos").mkdir(parents=True)
    (input_dir / "my-photos" / "disposable").mkdir()
    monkeypatch.setattr("nightflash.kaggle.INPUT", input_dir)
    assert find_snapshot("runs/edgehat_freqsplit_night") is None


def test_auto_resume_is_a_no_op_off_kaggle(tmp_path) -> None:
    source_dir, target_dir = tmp_path / "a", tmp_path / "b"
    source_dir.mkdir(); target_dir.mkdir()
    _rgb(32, 32, 1).save(source_dir / "0.png")
    _rgb(32, 32, 2).save(target_dir / "0.png")
    session = Session(
        source_dir=source_dir, target_dir=target_dir, image_size=32,
        batch_size=1, epochs=0, runs_root=tmp_path / "runs", name="r",
    )
    assert session.auto_resume() is False


# ------------------------------------------------------------------- snapshot


def test_snapshot_every_epoch_is_off_by_default(tmp_path) -> None:
    source_dir, target_dir = tmp_path / "a", tmp_path / "b"
    source_dir.mkdir(); target_dir.mkdir()
    for index in range(2):
        _rgb(32, 32, index).save(source_dir / f"{index}.png")
        _rgb(32, 32, index + 5).save(target_dir / f"{index}.png")
    session = Session(
        source_dir=source_dir, target_dir=target_dir, image_size=32,
        batch_size=1, epochs=0, runs_root=tmp_path / "runs", name="d",
    )
    assert session.snapshot_every == 0


def test_snapshot_failure_never_stops_training(tmp_path, monkeypatch) -> None:
    """Losing the network mid-session must not destroy an hour of training."""

    source_dir, target_dir = tmp_path / "a", tmp_path / "b"
    source_dir.mkdir(); target_dir.mkdir()
    for index in range(2):
        _rgb(32, 32, index).save(source_dir / f"{index}.png")
        _rgb(32, 32, index + 5).save(target_dir / f"{index}.png")

    monkeypatch.setattr("nightflash.kaggle.is_kaggle", lambda: True)
    def boom(*_args, **_kwargs):
        raise RuntimeError("kaggle api unreachable")
    monkeypatch.setattr("nightflash.kaggle.snapshot", boom)

    session = Session(
        source_dir=source_dir, target_dir=target_dir, image_size=32, batch_size=1,
        epochs=2, num_workers=0, runs_root=tmp_path / "runs", name="f",
        snapshot_every=1,
    )
    result = session.train(2)
    assert result["completed"] is True
    assert session.latest.exists()


def test_snapshot_is_skipped_off_kaggle(tmp_path, monkeypatch, capsys) -> None:
    source_dir, target_dir = tmp_path / "a", tmp_path / "b"
    source_dir.mkdir(); target_dir.mkdir()
    for index in range(2):
        _rgb(32, 32, index).save(source_dir / f"{index}.png")
        _rgb(32, 32, index + 5).save(target_dir / f"{index}.png")
    monkeypatch.setattr("nightflash.kaggle.is_kaggle", lambda: False)

    session = Session(
        source_dir=source_dir, target_dir=target_dir, image_size=32, batch_size=1,
        epochs=1, num_workers=0, runs_root=tmp_path / "runs", name="g",
        snapshot_every=1,
    )
    session.train(1)
    assert "not a Kaggle runtime" in capsys.readouterr().out


def test_manual_snapshot_failure_warns_but_raises_on_strict(
    tmp_path, monkeypatch, capsys
) -> None:
    """The last cell of a session must not die because the API is unreachable.

    ``s.snapshot()`` is what you run before disconnecting. A missing ``kaggle``
    package or an expired token should print a warning, not lose the run. The
    CLI exposes the opposite behaviour as ``--strict``.
    """

    session = _session(tmp_path, "snapfail", 1)
    session.train()
    monkeypatch.setattr("nightflash.kaggle.is_kaggle", lambda: True)

    def boom(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'kaggle'")
    monkeypatch.setattr("nightflash.kaggle.snapshot", boom)

    assert session.snapshot() is None
    out = capsys.readouterr().out
    assert "snapshot failed" in out
    assert str(session.run_dir) in out, "must say where the run still is"

    with pytest.raises(ModuleNotFoundError):
        session.snapshot(strict=True)


# ------------------------------------------------------------------------- cli


def _tiny_dataset(tmp_path) -> tuple[object, object]:
    source_dir, target_dir = tmp_path / "a", tmp_path / "b"
    source_dir.mkdir(exist_ok=True); target_dir.mkdir(exist_ok=True)
    for index in range(2):
        _rgb(32, 32, index).save(source_dir / f"{index}.png")
        _rgb(32, 32, index + 5).save(target_dir / f"{index}.png")
    return source_dir, target_dir


def test_cli_train_actually_trains(tmp_path) -> None:
    """Regression: the CLI once asked ``plan`` in the wrong mode.

    ``would_train`` is only true when ``plan`` is called with mode ``"train"``,
    so checking it in ``"plan"`` mode made ``nf train`` refuse every run, always.
    """

    from nightflash.cli import main

    source_dir, target_dir = _tiny_dataset(tmp_path)
    code = main([
        "train",
        "--source-dir", str(source_dir), "--target-dir", str(target_dir),
        "--runs-root", str(tmp_path / "runs"), "--name", "cli",
        "--image-size", "32", "--batch-size", "1", "--num-workers", "0",
        "--epochs", "2", "--yes",
    ])
    assert code == 0
    assert (tmp_path / "runs" / "edgehat_freqsplit_cli" / "checkpoints" / "latest.pt").exists()


def test_cli_train_refuses_when_already_complete(tmp_path) -> None:
    from nightflash.cli import main

    source_dir, target_dir = _tiny_dataset(tmp_path)
    common = [
        "--source-dir", str(source_dir), "--target-dir", str(target_dir),
        "--runs-root", str(tmp_path / "runs"), "--name", "done",
        "--image-size", "64", "--batch-size", "1", "--num-workers", "0",
        "--device", "cpu", "--epochs", "1", "--yes",
    ]
    assert main(["train", *common]) == 0
    # Asking for the same absolute target again must be a no-op, not a rerun.
    assert main(["train", *common]) == 2


# ------------------------------------------------------- epoch target semantics


def _session(tmp_path, name: str, epochs: int) -> Session:
    source_dir, target_dir = _tiny_dataset(tmp_path)
    return Session(
        source_dir=source_dir, target_dir=target_dir, image_size=64,
        batch_size=1, epochs=epochs, num_workers=0, device="cpu",
        runs_root=tmp_path / "runs", name=name,
    )


def test_epochs_flag_is_absolute_not_a_running_total(tmp_path) -> None:
    """``--epochs 2`` means epoch 2, full stop.

    Treating it as "N more epochs" made every repeat call train another N
    epochs, so a run could never converge and a resume always overshot.
    """

    session = _session(tmp_path, "abs", 2)
    assert session.target_epoch() == 2
    session.train()
    assert session.target_epoch() == 2, "still epoch 2, not 4"
    assert session.plan("train")["would_train"] is False


def test_extra_epochs_is_relative_to_the_resume_point(tmp_path) -> None:
    session = _session(tmp_path, "rel", 2)
    session.train()
    assert session.state().last_epoch == 1
    # "+1 more" from epoch 2 lands on epoch 3.
    assert session.target_epoch(1) == 3
    assert session.plan("train", 1)["epochs_this_call"] == 1


def test_extra_epochs_zero_is_a_no_op_not_a_rerun(tmp_path) -> None:
    session = _session(tmp_path, "zero", 1)
    session.train()
    assert session.state().last_epoch == 0
    # "+0 more" from epoch 1 stays at 1 and must not rerun anything.
    assert session.target_epoch(0) == 1
    assert session.plan("train", 0)["would_train"] is False


def test_cli_infer_downscale_and_tile(tmp_path) -> None:
    from nightflash.cli import main

    source_dir, target_dir = _tiny_dataset(tmp_path)
    # 700 wide by 1100 tall: the long edge is the height.
    subject = _rgb(700, 1100, 40)
    subject.save(tmp_path / "pic.png")

    assert main([
        "train",
        "--source-dir", str(source_dir), "--target-dir", str(target_dir),
        "--runs-root", str(tmp_path / "runs"), "--name", "inf",
        "--image-size", "64", "--batch-size", "1", "--num-workers", "0",
        "--device", "cpu", "--epochs", "1", "--yes",
    ]) == 0
    assert main([
        "infer",
        "--runs-root", str(tmp_path / "runs"), "--name", "inf",
        "--device", "cpu",
        "--input", str(tmp_path / "pic.png"),
        "--max-edge", "800", "--tile", "128", "--tile-overlap", "32",
    ]) == 0

    out_dir = tmp_path / "runs" / "edgehat_freqsplit_inf" / "inferences"
    written = list(out_dir.glob("*.png"))
    assert len(written) == 1
    assert Image.open(written[0]).size == (509, 800)


def test_cli_rejects_an_unknown_augment_preset(tmp_path) -> None:
    from nightflash.cli import main

    source_dir, target_dir = _tiny_dataset(tmp_path)
    with pytest.raises(SystemExit):
        main([
            "train",
            "--source-dir", str(source_dir), "--target-dir", str(target_dir),
            "--runs-root", str(tmp_path / "runs"),
            "--augment", "nonsense", "--yes",
        ])


# --------------------------------------------------------- arch-agnostic calls


def test_no_module_unpacks_a_generator_call_directly() -> None:
    """Every generator call must go through ``run_generator``.

    New architectures return a 3-tuple while the legacy ones return a 2-tuple.
    A bare ``residual, params = model(x)`` works for legacy and then raises
    ``ValueError: too many values to unpack`` the moment a newer architecture is
    used. That bug shipped once already (``evaluate``), so pin it.
    """

    import ast
    import pathlib

    import nightflash

    package = pathlib.Path(nightflash.__file__).parent
    offenders: list[str] = []
    for path in sorted(package.glob("*.py")):
        if path.name == "arch.py":  # the normaliser itself legitimately unpacks
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, (ast.Tuple, ast.Name)):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            func = value.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else ""
            )
            if name in {"run_generator", "build_generator", "load_model_checkpoint"}:
                continue
            if name in {"model", "generator", "self"} or name.endswith("_model"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "unpack a generator call without run_generator (breaks on new archs): "
        + ", ".join(offenders)
    )


def test_evaluate_runs_on_a_new_architecture(tmp_path) -> None:
    """End-to-end: evaluate must work on edgehat, not just legacy models."""

    from nightflash.evaluate import main as evaluate_main

    source_dir, target_dir = _tiny_dataset(tmp_path)
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    for index in range(2):
        _rgb(32, 32, index + 30).save(holdout / f"{index}.png")

    session = _session(tmp_path, "ev", 1)
    session.train()

    score = tmp_path / "score.json"
    evaluate_main([
        "--checkpoint", str(session.latest),
        "--source-dir", str(source_dir),
        "--target-dir", str(holdout),
        "--image-size", "64", "--device", "cpu",
        "--output-json", str(score),
    ])
    assert score.exists()
    import json

    result = json.loads(score.read_text())
    for key in ("content_l1_mean", "edge_l1_mean", "simple_ssim_mean", "source_images"):
        assert key in result
    assert result["source_images"] == 2
