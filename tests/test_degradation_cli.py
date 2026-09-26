"""CLI surface for the training modes, degradation and validation options."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nightflash.cli import build_parser, main


def _pool(root: Path, n: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        g = np.random.default_rng(i)
        a = g.random((96, 96, 3)) * 0.3
        yy, xx = np.mgrid[0:96, 0:96]
        d = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
        a[d < 24] = 0.85
        Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(root / f"n{i}.png")
    return root


def _base(tmp_path, name: str, pool: Path, *extra: str) -> list[str]:
    return [
        "train", "--target-dir", str(pool), "--runs-root", str(tmp_path / "runs"),
        "--name", name, "--image-size", "64", "--batch-size", "2",
        "--num-workers", "0", "--device", "cpu", "--yes", "--val-count", "2",
        *extra,
    ]


# ------------------------------------------------------------------ flag shape


def test_new_flags_are_registered() -> None:
    parser = build_parser()
    import argparse

    sub = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    options: set[str] = set()
    for command in sub.choices:
        for action in sub.choices[command]._actions:
            options.update(action.option_strings)
    for flag in (
        "--train-mode", "--degrade-profile", "--degrade-none-prob",
        "--no-degrade-curriculum", "--degrade-config", "--paired-weight",
        "--degrade-contrast-min", "--degrade-contrast-max",
        "--degrade-resize-min", "--degrade-noise-max", "--degrade-wb-min",
        "--val-every", "--val-count", "--val-select", "--val-cols",
        "--val-split", "--val-holdout-dir", "--val-seed", "--val-layout",
        "--no-keep-best", "--keep-last", "--ema-decay",
    ):
        assert flag in options, f"{flag} missing from the CLI"


def test_degrade_stage_pairs_have_min_and_max() -> None:
    parser = build_parser()
    import argparse

    sub = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    options = {o for a in sub.choices["train"]._actions for o in a.option_strings}
    for stage in ("exposure", "black-lift", "contrast", "saturation",
                  "wb", "resize", "blur", "noise"):
        assert f"--degrade-{stage}-min" in options
        assert f"--degrade-{stage}-max" in options


def test_curriculum_is_on_by_default(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 4)
    assert main(_base(tmp_path, "cd", pool, "--epochs", "1")) == 0
    saved = json.loads((tmp_path / "runs" / "edgehat_freqsplit_cd" / "config.json").read_text())
    assert saved["degrade"]["curriculum"] is True

    assert main(_base(tmp_path, "co", pool, "--epochs", "1",
                      "--no-degrade-curriculum")) == 0
    saved = json.loads((tmp_path / "runs" / "edgehat_freqsplit_co" / "config.json").read_text())
    assert saved["degrade"]["curriculum"] is False


def test_config_file_beats_an_unspecified_flag(tmp_path) -> None:
    """A --degrade-config value must survive when the flag was not given.

    argparse cannot distinguish "not passed" from "passed the default" unless the
    default is None, which is why these two flags default to None.
    """

    path = tmp_path / "d.json"
    path.write_text(json.dumps({"none_prob": 0.33, "curriculum": False}))
    pool = _pool(tmp_path / "pool", 4)
    assert main(_base(tmp_path, "cdf", pool, "--epochs", "1",
                      "--degrade-config", str(path))) == 0
    saved = json.loads((tmp_path / "runs" / "edgehat_freqsplit_cdf" / "config.json").read_text())
    assert saved["degrade"]["none_prob"] == 0.33
    assert saved["degrade"]["curriculum"] is False

    # an explicit flag must still win
    assert main(_base(tmp_path, "cdg", pool, "--epochs", "1",
                      "--degrade-config", str(path),
                      "--degrade-none-prob", "0.05")) == 0
    saved = json.loads((tmp_path / "runs" / "edgehat_freqsplit_cdg" / "config.json").read_text())
    assert saved["degrade"]["none_prob"] == 0.05


def test_val_dir_and_holdout_dir_are_distinct() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "train", "--target-dir", "x",
        "--val-dir", "/legacy", "--val-holdout-dir", "/held",
    ])
    assert args.val_dir == "/held"
    assert args.source_dir is None


def test_degrade_config_json_is_applied(tmp_path) -> None:
    path = tmp_path / "degrade.json"
    path.write_text(json.dumps({"resize_min": 0.11, "resize_max": 0.22,
                                "none_prob": 0.33}))
    pool = _pool(tmp_path / "pool", 4)
    assert main(_base(tmp_path, "cfg", pool, "--epochs", "1",
                      "--degrade-config", str(path))) == 0
    saved = json.loads((tmp_path / "runs" / "edgehat_freqsplit_cfg" / "config.json").read_text())
    assert saved["degrade"]["resize_min"] == 0.11
    assert saved["degrade"]["none_prob"] == 0.33


def test_stage_overrides_reach_the_config(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 4)
    assert main(_base(tmp_path, "ovr", pool, "--epochs", "1",
                      "--degrade-contrast-min", "0.31", "--degrade-contrast-max", "0.42",
                      "--degrade-resize-min", "0.43", "--degrade-noise-max", "0.002")) == 0
    saved = json.loads((tmp_path / "runs" / "edgehat_freqsplit_ovr" / "config.json").read_text())
    assert saved["degrade"]["contrast_min"] == 0.31
    assert saved["degrade"]["contrast_max"] == 0.42
    assert saved["degrade"]["resize_min"] == 0.43
    assert saved["degrade"]["noise_max"] == 0.002


# ------------------------------------------------------------------- behaviour


def test_cli_default_is_paired_and_writes_a_strip(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 6)
    assert main(_base(tmp_path, "d", pool, "--epochs", "1")) == 0
    run = tmp_path / "runs" / "edgehat_freqsplit_d"
    saved = json.loads((run / "config.json").read_text())
    assert saved["train_mode"] == "paired"
    assert saved["degrade"]["curriculum"] is True
    assert list((run / "validations").glob("epoch-*.png"))
    assert (run / "validations" / "history.csv").exists()
    assert (run / "checkpoints" / "best.pt").exists()


def test_cli_sameset_arm(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 6)
    assert main(_base(tmp_path, "ss", pool, "--epochs", "1",
                      "--train-mode", "sameset", "--degrade-profile", "strong")) == 0
    run = tmp_path / "runs" / "edgehat_freqsplit_ss"
    assert json.loads((run / "config.json").read_text())["train_mode"] == "sameset"
    assert not (run / "checkpoints" / "best.pt").exists()


def test_cli_two_arms_on_two_names_do_not_collide(tmp_path) -> None:
    """paired on one 'GPU', sameset on the other, in the same runs root."""

    pool = _pool(tmp_path / "pool", 8)
    assert main(_base(tmp_path, "armA", pool, "--epochs", "1")) == 0
    assert main(_base(tmp_path, "armB", pool, "--epochs", "1",
                      "--train-mode", "sameset")) == 0
    root = tmp_path / "runs"
    assert (root / "edgehat_freqsplit_armA" / "checkpoints" / "latest.pt").exists()
    assert (root / "edgehat_freqsplit_armB" / "checkpoints" / "latest.pt").exists()
    assert (root / "edgehat_freqsplit_armA" / "validations" / "history.csv").exists()
    assert (root / "edgehat_freqsplit_armB" / "validations" / "history.csv").exists()


def test_cli_keep_last_prunes(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 6)
    assert main(_base(tmp_path, "k", pool, "--epochs", "3", "--keep-last", "2")) == 0
    kept = sorted(p.name for p in (tmp_path / "runs" / "edgehat_freqsplit_k"
                                   / "checkpoints").glob("epoch-*.pt"))
    assert len(kept) == 2, kept


def test_cli_no_keep_best(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 6)
    assert main(_base(tmp_path, "nb", pool, "--epochs", "1", "--no-keep-best")) == 0
    assert not (tmp_path / "runs" / "edgehat_freqsplit_nb"
                / "checkpoints" / "best.pt").exists()


def test_cli_val_every_zero_disables_strips(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 6)
    assert main(_base(tmp_path, "ve", pool, "--epochs", "2", "--val-every", "0")) == 0
    run = tmp_path / "runs" / "edgehat_freqsplit_ve"
    assert not list((run / "validations").glob("epoch-*.png"))


def test_cli_val_select_random_still_writes_a_strip(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 8)
    assert main(_base(tmp_path, "vr", pool, "--epochs", "1",
                      "--val-select", "random")) == 0
    run = tmp_path / "runs" / "edgehat_freqsplit_vr"
    assert list((run / "validations").glob("epoch-*.png"))
    saved = json.loads((run / "config.json").read_text())
    assert saved["validation"]["select"] == "random"


def test_cli_holdout_dir_overrides_split(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 8)
    held = _pool(tmp_path / "held", 2)
    assert main(_base(tmp_path, "ho", pool, "--epochs", "1",
                      "--val-holdout-dir", str(held), "--val-split", "0.9")) == 0
    saved = json.loads((tmp_path / "runs" / "edgehat_freqsplit_ho" / "config.json").read_text())
    assert saved["validation"]["dir"] == str(held)


def test_cli_ema_and_cols_and_layout(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 6)
    assert main(_base(tmp_path, "e", pool, "--epochs", "1", "--ema-decay", "0.9",
                      "--val-cols", "2", "--val-layout", "cols")) == 0
    import torch

    run = tmp_path / "runs" / "edgehat_freqsplit_e"
    checkpoint = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu",
                            weights_only=False)
    assert checkpoint.get("ema")
    strip = sorted((run / "validations").glob("epoch-*.png"))[0]
    assert strip.stat().st_size > 0


def test_cli_paired_weight_zero_disables_supervision(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 6)
    assert main(_base(tmp_path, "w", pool, "--epochs", "1", "--paired-weight", "0.0")) == 0
    saved = json.loads((tmp_path / "runs" / "edgehat_freqsplit_w" / "config.json").read_text())
    assert saved["paired_weight"] == 0.0


def test_cli_config_reports_the_new_settings(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 4)
    main(_base(tmp_path, "c", pool, "--epochs", "1"))
    assert main(["config", "--runs-root", str(tmp_path / "runs"), "--name", "c"]) == 0


def test_cli_rejects_an_unknown_profile(tmp_path) -> None:
    pool = _pool(tmp_path / "pool", 4)
    with pytest.raises(SystemExit):
        main(_base(tmp_path, "bad", pool, "--degrade-profile", "nonsense"))
