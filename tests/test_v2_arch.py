"""Tests for the v2 architectures, heads, session and briefing."""

from __future__ import annotations

import pytest
import torch

from nightflash.arch import (
    ARCHITECTURES,
    HEADS,
    build_generator,
    new_arch_config,
    run_generator,
)
from nightflash.briefing import brief
from nightflash.heads import FreqSplitHead, Lut3DHead, PerPixelHead, TanhHead
from nightflash.session import Session

NEW_ARCHS = ("edgehat", "swinir")
SIZES = ((32, 32), (64, 64), (48, 80), (96, 64), (75, 100))


# ---------------------------------------------------------------- registry


def test_registry_covers_everything() -> None:
    from nightflash.heads import ALL_HEADS, SIMPLE_HEADS

    assert set(ARCHITECTURES) == {"edgehat", "swinir", "hybrid", "restormer"}
    assert set(SIMPLE_HEADS) == {"tanh", "perpixel", "freqsplit", "lut3d"}
    assert set(HEADS) == set(ALL_HEADS)
    assert "lut3d+freqsplit" in HEADS


@pytest.mark.parametrize("arch", NEW_ARCHS)
@pytest.mark.parametrize("head", HEADS)
def test_every_arch_head_forward_backward(arch: str, head: str) -> None:
    config = new_arch_config(arch, head=head)
    model = build_generator(config)
    for height, width in SIZES:
        x = torch.rand(1, config.in_channels, height, width)
        residual, effects, extras = run_generator(model, x)
        assert residual.shape == (1, 3, height, width)
        assert effects.shape == (1, config.effect_dim)
        for value in extras.values():
            if value is not None:
                assert value.shape[0] == 1
    (residual.mean() + effects.mean()).backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_run_generator_normalises_legacy_return(arch: str) -> None:
    """hybrid returns a 2-tuple; run_generator must hide that difference."""

    config = new_arch_config(arch)
    model = build_generator(config)
    residual, effects, extras = run_generator(model, torch.rand(1, config.in_channels, 64, 64))
    assert residual.shape[1] == 3
    assert effects.shape[1] == config.effect_dim
    assert set(extras) == {"subject", "region", "halation"}


def test_unknown_arch_raises() -> None:
    from nightflash.config import ModelConfig

    with pytest.raises(ValueError, match="unknown arch"):
        build_generator(ModelConfig(arch="does-not-exist"))


# ------------------------------------------------------------------ heads


def test_tanh_head_is_bounded() -> None:
    head = TanhHead(16, scale=0.35)
    out = head(torch.randn(2, 16, 32, 32) * 50)
    assert out.abs().max() <= 0.35 + 1e-6


def test_perpixel_head_is_bounded_by_limit() -> None:
    head = PerPixelHead(16, limit=0.5)
    out = head(torch.randn(2, 16, 32, 32) * 50)
    assert out.abs().max() <= 0.5 + 1e-6


def test_freqsplit_starts_as_identity() -> None:
    """Zero-init means a fresh head contributes nothing, so early training is stable."""

    head = FreqSplitHead(16)
    assert head(torch.randn(2, 16, 32, 32)).abs().max() < 1e-6


def test_freqsplit_bands_are_separable() -> None:
    head = FreqSplitHead(16)
    features = torch.randn(2, 16, 32, 32)
    bands = head.band_magnitudes(features)
    assert set(bands) == {"low", "mid", "high"}
    for value in bands.values():
        assert value.ndim == 0


def test_lut3d_identity_is_exact() -> None:
    head = Lut3DHead(8, size=17).eval()
    image = torch.rand(2, 3, 24, 32)
    with torch.no_grad():
        graded = head._apply_table(image)
    assert torch.allclose(graded, image, atol=1e-5)


def test_lut3d_applies_a_known_transform() -> None:
    head = Lut3DHead(8, size=9).eval()
    size = 9
    ramp = torch.linspace(0, 1, size)
    ri = torch.arange(size).view(size, 1, 1).expand(size, size, size)
    gi = torch.arange(size).view(1, size, 1).expand(size, size, size)
    bi = torch.arange(size).view(1, 1, size).expand(size, size, size)
    with torch.no_grad():  # blue channel takes the value of red
        head.table.copy_(
            torch.stack([ramp[ri], ramp[gi], ramp[ri]], dim=-1).reshape(-1, 3)
        )
    image = torch.rand(1, 3, 12, 12)
    with torch.no_grad():
        graded = head._apply_table(image)
    assert torch.allclose(graded[:, 2], image[:, 0], atol=1e-5)
    assert torch.allclose(graded[:, 1], image[:, 1], atol=1e-5)


def test_lut3d_requires_image() -> None:
    head = Lut3DHead(8, size=9)
    with pytest.raises(ValueError, match="needs the source image"):
        head(torch.zeros(1, 8, 8, 8))


# ----------------------------------------------------------- capabilities


def test_capabilities_are_independent() -> None:
    base = dict(arch="edgehat", head="freqsplit")
    full = new_arch_config(**base, use_halation=True, use_subject_gate=True, use_region_exposure=True)
    none = new_arch_config(**base, use_halation=False, use_subject_gate=False, use_region_exposure=False)
    assert set(full.capability_names()) == {"halation", "subject-gate", "region-exposure"}
    assert none.capability_names() == []


@pytest.mark.parametrize(
    "halation,subject,region",
    [(True, False, False), (False, True, False), (False, False, True), (True, True, True)],
)
def test_capability_subsets_run(halation: bool, subject: bool, region: bool) -> None:
    config = new_arch_config(
        "edgehat", head="freqsplit",
        use_halation=halation, use_subject_gate=subject, use_region_exposure=region,
    )
    model = build_generator(config)
    _, _, extras = run_generator(model, torch.rand(1, 3, 64, 64))
    assert (extras["halation"] is not None) is halation
    assert (extras["subject"] is not None) is subject
    assert (extras["region"] is not None) is region


def test_extras_start_disabled() -> None:
    """Zero-initialised mix scalars mean the extras cannot destabilise step one."""

    config = new_arch_config("edgehat", head="freqsplit")
    model = build_generator(config)
    assert float(model.subject_mix.abs()) < 1e-8
    assert float(model.region_mix.abs()) < 1e-8
    assert float(model.halation_mix.abs()) < 1e-8


def test_halation_responds_only_above_threshold() -> None:
    """The branch is zero-initialised, so test the physics, not the weights.

    Give the spill convolution a non-zero output filter, then the glow must appear
    for bright input and stay absent for input below the highlight threshold.
    """

    from nightflash.edgehat import HalationBranch

    branch = HalationBranch(16, threshold=0.7, strength=0.5, tint=(1.0, 0.6, 0.3))
    with torch.no_grad():
        # These convs are built with bias=False, so only the weights exist. Make
        # both positive so the highlight map survives the ReLU.
        branch.spill[0].weight.fill_(1.0)
        branch.spill[-1].weight.fill_(1.0)
        branch.tonal[-1].weight.zero_()

    dark = torch.full((1, 3, 32, 32), 0.2)
    bright = torch.full((1, 3, 32, 32), 0.95)
    features = torch.zeros(1, 16, 32, 32)

    assert float(branch(dark, features).abs().max()) < 1e-6, "glow leaked into shadows"
    bright_glow = branch(bright, features)
    assert float(bright_glow.abs().max()) > 0.0, "no glow on a highlight"
    # The tint is warm: red leads blue.
    assert bright_glow[0, 0].mean() > bright_glow[0, 2].mean()


@pytest.mark.parametrize(
    "luma,expect_glow", [(0.2, False), (0.6, False), (0.71, True), (0.95, True)]
)
def test_halation_threshold_is_respected(luma: float, expect_glow: bool) -> None:
    from nightflash.edgehat import HalationBranch

    branch = HalationBranch(16, threshold=0.7, strength=0.5, tint=(1.0, 0.6, 0.3))
    with torch.no_grad():
        # Make the first conv strictly positive so a positive highlight map cannot
        # be annihilated by random negative weights.
        branch.spill[0].weight.fill_(1.0)
        branch.spill[-1].weight.fill_(1.0)
    out = branch(torch.full((1, 3, 16, 16), luma), torch.zeros(1, 16, 16, 16))
    assert (float(out.abs().max()) > 0.0) is expect_glow


def test_halation_starts_as_noop() -> None:
    from nightflash.edgehat import HalationBranch

    branch = HalationBranch(16, threshold=0.7, strength=0.5, tint=(1.0, 0.6, 0.3))
    bright = torch.full((1, 3, 16, 16), 1.0)
    features = torch.zeros(1, 16, 16, 16)
    assert float(branch(bright, features).abs().max()) < 1e-6


def test_region_exposure_starts_as_noop_then_varies() -> None:
    """Zero-init keeps it inert at step 1; once active it varies per region."""

    from nightflash.edgehat import RegionExposure

    head = RegionExposure(32, grid=4)
    features = torch.zeros(1, 32, 32, 32)
    features[..., :16, :16] = 1.0  # bright top-left quadrant, dark elsewhere
    assert float(head(features).abs().max()) < 1e-6, "should be inert when zero-initialised"

    with torch.no_grad():  # activate the projection
        head.to_coeff.weight.fill_(0.1)
    out = head(features)
    assert out.shape == (1, 3, 32, 32)
    top_left = float(out[..., 2, 2].mean())
    bottom_right = float(out[..., -3, -3].mean())
    assert abs(top_left - bottom_right) > 1e-4, "regions should differ"


def test_subject_gate_defaults_to_background() -> None:
    from nightflash.edgehat import SubjectGate

    gate = SubjectGate(32)
    value = gate(torch.zeros(1, 32, 16, 16))
    assert float(value.mean()) < 0.25  # sigmoid(-2) ~ 0.12


def test_subject_gate_fuses_external_mask() -> None:
    from nightflash.edgehat import SubjectGate

    gate = SubjectGate(32)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., :8, :8] = 1.0
    out = gate(torch.zeros(1, 32, 16, 16), mask)
    assert out[0, 0, 0, 0] > out[0, 0, -1, -1]


def test_mask_input_changes_channel_count() -> None:
    config = new_arch_config("edgehat", head="freqsplit", use_mask=True)
    model = build_generator(config)
    assert config.in_channels == 4
    residual, _, _ = run_generator(model, torch.rand(1, 4, 64, 64), torch.rand(1, 1, 64, 64))
    assert residual.shape == (1, 3, 64, 64)


# --------------------------------------------------------------- session


def _tiny_session(tmp_path, **kwargs) -> Session:
    defaults = dict(
        arch="edgehat", head="freqsplit", name="t",
        runs_root=tmp_path / "runs",
        image_size=32, batch_size=2, epochs=1, device="cpu",
        use_amp=False, num_workers=0, preview_every=0, log_every=1,
        halation=False, subject_gate=False, region_exposure=False,
    )
    defaults.update(kwargs)
    return Session(**defaults)


def test_session_isolates_architectures(tmp_path) -> None:
    a = _tiny_session(tmp_path, arch="edgehat")
    b = _tiny_session(tmp_path, arch="swinir")
    assert a.run_dir != b.run_dir
    assert "edgehat" in a.run_dir.name and "swinir" in b.run_dir.name


def test_session_isolates_heads(tmp_path) -> None:
    a = _tiny_session(tmp_path, head="freqsplit")
    b = _tiny_session(tmp_path, head="tanh")
    assert a.run_dir != b.run_dir


def test_state_before_any_training(tmp_path) -> None:
    session = _tiny_session(tmp_path)
    state = session.state()
    assert state.has_checkpoint is False
    assert "NEW" in state.status or "EMPTY" in state.status


def test_target_epoch_is_absolute(tmp_path) -> None:
    session = _tiny_session(tmp_path, epochs=10)
    assert session.target_epoch() == 10  # fresh: 0 + 10


def test_plan_writes_nothing(tmp_path) -> None:
    session = _tiny_session(tmp_path)
    before = sorted(p.name for p in session.run_dir.iterdir())
    session.plan("plan")
    session.plan("train")
    assert sorted(p.name for p in session.run_dir.iterdir()) == before


def test_plan_reports_infer_without_training(tmp_path) -> None:
    session = _tiny_session(tmp_path)
    plan = session.plan("infer")
    assert plan["would_train"] is False
    assert "INFER" in plan["action"]


def test_briefing_contains_all_sections(tmp_path) -> None:
    session = _tiny_session(tmp_path, source_dir=tmp_path / "src", target_dir=tmp_path / "tgt")
    text = brief(session, mode="plan")
    for heading in (
        "1. WHAT THIS MODEL IS",
        "2. CAPABILITIES",
        "3. WHAT WILL BE TRAINED",
        "4. WHERE EVERYTHING GOES",
        "5. WHAT WILL HAPPEN",
        "6. CURRENT STATE",
        "7. DECISION",
    ):
        assert heading in text, f"briefing is missing section {heading}"
    assert str(session.run_dir) in text
    assert session.config.arch in text
    assert session.config.head in text


def test_briefing_never_offers_a_noop_continue(tmp_path) -> None:
    session = _tiny_session(tmp_path, epochs=1)
    session.train.__doc__  # noqa: B018 - readability
    # Simulate a finished run by writing a checkpoint with the final epoch.
    import torch

    from nightflash.arch import build_generator

    model = build_generator(session.config)
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": torch.zeros(1),
            "optimizer_g": torch.optim.AdamW(model.parameters()).state_dict(),
            "optimizer_d": torch.optim.AdamW(model.parameters()).state_dict(),
            "model_config": session.config.to_dict(),
            "epoch": 0, "global_step": 1,
        },
        session.latest,
    )
    text = brief(session, mode="plan")
    assert "would train zero epochs" in text
    assert "extra_epochs=50" in text


def test_reset_archives_by_default(tmp_path) -> None:
    session = _tiny_session(tmp_path)
    (session.checkpoint_dir / "latest.pt").write_bytes(b"x")
    destination = session.reset()
    assert destination is not None
    assert destination.name.endswith("_archive_1")
    assert destination.exists()
    assert (destination / "checkpoints" / "latest.pt").exists()
    assert not session.latest.exists()


def test_reset_can_delete_in_place(tmp_path) -> None:
    session = _tiny_session(tmp_path)
    (session.checkpoint_dir / "latest.pt").write_bytes(b"x")
    destination = session.reset(archive=False)
    assert destination is None
    assert not session.latest.exists()
    assert not list(tmp_path.glob("runs/*_archive_*"))


def test_noop_guard_refuses_to_train(tmp_path) -> None:
    session = _tiny_session(tmp_path)
    import torch

    from nightflash.arch import build_generator

    model = build_generator(session.config)
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": torch.zeros(1),
            "optimizer_g": torch.optim.AdamW(model.parameters()).state_dict(),
            "optimizer_d": torch.optim.AdamW(model.parameters()).state_dict(),
            "model_config": session.config.to_dict(),
            "epoch": 4, "global_step": 10,
        },
        session.latest,
    )
    result = session.train(extra_epochs=0)
    assert result["epochs_this_call"] == 0
    assert "NO-OP GUARD" in result["action"]


def test_state_flags_arch_mismatch(tmp_path) -> None:
    session = _tiny_session(tmp_path, arch="edgehat")
    import torch

    from nightflash.arch import build_generator

    other = build_generator(new_arch_config("swinir"))
    torch.save(
        {
            "model": other.state_dict(),
            "discriminator": torch.zeros(1),
            "optimizer_g": torch.optim.AdamW(other.parameters()).state_dict(),
            "optimizer_d": torch.optim.AdamW(other.parameters()).state_dict(),
            "model_config": new_arch_config("swinir").to_dict(),
            "epoch": 1, "global_step": 2,
        },
        session.latest,
    )
    state = session.state()
    assert state.issue is not None
    assert "BLOCKED" in state.status
