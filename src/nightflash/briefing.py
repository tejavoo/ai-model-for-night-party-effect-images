"""Pre-flight briefing.

Prints everything you need to know *before* a run touches the GPU: what the
flags actually selected, where every artifact will land, what will happen step by
step, what state the run directory is in, and a decision menu with the exact
call for each choice.

The text is generated from the live config and the live filesystem, so it cannot
drift away from what the code will really do.
"""

from __future__ import annotations

import shutil
from typing import Any, Iterable, Sequence

from .arch import (
    ARCHITECTURES,
    ARCH_DESCRIPTIONS,
    CAPABILITIES,
    HEADS,
    HEAD_DESCRIPTIONS,
)
from .config import ModelConfig
from .session import Session

WIDTH = 78
CHECK = "✓"
CROSS = "✗"
DOT = "·"


def rule(char: str = "─") -> str:
    return char * WIDTH


def heading(title: str) -> str:
    return f"\n{rule()}\n {title}\n{rule()}"


def wrap(text: str, indent: str = "  ") -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=WIDTH - len(indent), initial_indent=indent,
                         subsequent_indent=indent)


def kv(key: str, value: Any, width: int = 26) -> str:
    return f"  {key:<{width}} {value}"


def yes_no(flag: bool) -> str:
    return f"{CHECK} on" if flag else f"{CROSS} off"


def _terminal_width() -> int:
    try:
        return max(60, min(100, shutil.get_terminal_size().columns))
    except Exception:
        return 80


def architecture_table() -> str:
    lines = [
        kv("architecture", ""),
        f"    {'edgehat':<10} {ARCH_DESCRIPTIONS['edgehat']}",
        f"    {'swinir':<10} {ARCH_DESCRIPTIONS['swinir']}",
        f"    {'restormer':<10} {ARCH_DESCRIPTIONS['restormer']}",
        f"    {'hybrid':<10} {ARCH_DESCRIPTIONS['hybrid']}",
    ]
    return "\n".join(lines)


def head_table() -> str:
    return "\n".join(
        f"    {'tanh':<10} {HEAD_DESCRIPTIONS['tanh']}\n"
        f"    {'perpixel':<10} {HEAD_DESCRIPTIONS['perpixel']}\n"
        f"    {'freqsplit':<10} {HEAD_DESCRIPTIONS['freqsplit']}\n"
        f"    {'lut3d':<10} {HEAD_DESCRIPTIONS['lut3d']}"
        for _ in (0,)
    )


def _capability_lines(config: ModelConfig) -> list[str]:
    lines: list[str] = []
    lines.append(
        kv("halation", yes_no(config.use_halation)
           + "   thresholded luminance, dilated depthwise spread, warm tint")
    )
    lines.append(
        kv("subject-gate", yes_no(config.use_subject_gate)
           + "   soft S(x,y) map; guided by a Vision mask when supplied")
    )
    lines.append(
        kv("region-exposure", yes_no(config.use_region_exposure)
           + f"   {config.region_grid}x{config.region_grid} coefficients, bilinearly upsampled")
    )
    return lines


def _flow_lines(session: Session, plan: dict[str, Any]) -> list[str]:
    state = plan["state"]
    lines = []
    step = 1

    lines.append(f"  {step}. validate dataset paths and count images")
    if session.source_dir:
        lines.append(f"       source : {session.source_dir}")
    if session.target_dir:
        lines.append(f"       target : {session.target_dir}")
    if session.val_dir:
        lines.append(f"       val    : {session.val_dir}")
    if session.mask_dir:
        lines.append(f"       masks  : {session.mask_dir}")
    else:
        lines.append("       masks  : none (subject gate will learn from appearance)")
    step += 1

    if plan["would_train"]:
        lines.append(f"  {step}. build {session.config.arch} and open a PatchGAN discriminator")
        step += 1
        if state["has_checkpoint"] and session.resume_enabled:
            lines.append(
                f"  {step}. resume from latest.pt (epoch {state['last_epoch']}, "
                f"step {state['global_step']}), restore AdamW moments"
            )
        else:
            lines.append(f"  {step}. start fresh: random init, no warm transfer")
        step += 1
        lines.append(
            f"  {step}. train epochs {plan['start_epoch'] + 1}..{plan['target_epoch']} "
            f"({plan['epochs_this_call']} epoch(s)), {session.image_size}px, "
            f"batch {session.batch_size}"
        )
        step += 1
        lines.append("       per step: discriminator update, then generator update")
        lines.append("       losses  : PatchGAN + 1.5 content/edge + 0.35 moment "
                     "+ 0.75 identity + 0.02 effect")
        step += 1
        lines.append(
            f"  {step}. every {session.preview_every} steps write a "
            "source | generated | target preview and a checkpoint"
        )
        step += 1
        lines.append(f"  {step}. write config.json and {session.checkpoint_dir}/latest.pt")
        step += 1
    else:
        lines.append(f"  {step}. load latest.pt and evaluate it (no training)")
        step += 1

    lines.append(f"  {step}. inference writes to {session.inference_dir}/")
    step += 1
    lines.append(
        f"       whole frame (default) or s.infer(..., tile=512) for native-resolution detail"
    )
    return lines


def _paths_block(session: Session) -> str:
    return "\n".join(
        [
            kv("run directory", session.run_dir),
            kv("  config", session.run_dir / "config.json"),
            kv("  checkpoints", session.checkpoint_dir / "latest.pt"),
            kv("  previews", session.preview_dir),
            kv("  inferences", session.inference_dir),
            kv("  evaluations", session.run_dir / "evaluations"),
            kv("  logs", session.log_path),
        ]
    )


def _state_block(session: Session) -> str:
    state = session.state()
    lines = [
        kv("status", state.status),
        kv("  arch / head", f"{state.saved_arch or '-'} / {state.saved_head or '-'}"),
        kv("  epoch", state.last_epoch if state.last_epoch is not None else "-"),
        kv("  global step", state.global_step if state.global_step is not None else "-"),
        kv("  checkpoint", f"{state.checkpoint_bytes / 1e6:.1f} MB" if state.checkpoint_bytes else "-"),
        kv("  previews", state.preview_count),
        kv("  inferences", state.inference_count),
        kv("  warm start", "present" if state.has_warm_start else "none"),
    ]
    if state.issue:
        lines.append(kv("  ISSUE", state.issue))
    return "\n".join(lines)


def _decision_block(session: Session, plan: dict[str, Any]) -> str:
    state = plan["state"]
    lines: list[str] = []
    blocked = bool(state["issue"])
    start = plan["start_epoch"] + 1
    target = plan["target_epoch"]
    goal = target

    if blocked:
        lines.append(f"  {CROSS} BLOCKED. Fix the issue above, or pick a different name:")
        lines.append("")
        lines.append("      s = Session(arch=..., head=..., name='other', ...)")
        return "\n".join(lines)

    if state["has_checkpoint"]:
        if goal <= start:
            lines.append(
                f"  [!] The default target ({target}) is at or before the resume point "
                f"({start})."
            )
            lines.append("      Continuing with the default would train zero epochs, so")
            lines.append("      s.train() will refuse. Pick a positive extra-epoch count:")
            lines.append("")
            lines.append("  [1] CONTINUE  resume from epoch " + str(start) + ", add 50 epochs")
            lines.append("      s.train(extra_epochs=50)")
        else:
            lines.append(
                f"  [1] CONTINUE  resume from epoch {start}, train to {target} "
                f"({target - start} epoch(s))"
            )
            lines.append("      s.train()")
            lines.append("")
            lines.append("  [2] CONTINUE further  resume and add 50 more epochs")
            lines.append("      s.train(extra_epochs=50)")
            lines.append("")
        lines.append("")
        lines.append("  [3] INFER ONLY  load the checkpoint, skip training entirely")
        lines.append("      s.infer('photo.jpg', seeds=3)")
        lines.append("      s.infer('photo.jpg', tile=512)   # full-resolution detail")
        lines.append("")
        lines.append("  [4] PLAN ONLY  print what would happen, change nothing")
        lines.append("      s.plan('plan')")
        lines.append("")
        lines.append("  [5] START FRESH  archive the current run, retrain from zero")
        lines.append("      s.reset()          # moves this run to <name>_archive_1 first")
        lines.append("      s.train()")
        lines.append("")
        lines.append("  [6] DELETE AND START FRESH  no archive, wipe in place")
        lines.append("      s.reset(archive=False)")
        lines.append("      s.train()")
    else:
        lines.append(f"  [1] START FRESH  train epochs 1..{target}")
        lines.append("      s.train()")
        lines.append("")
        lines.append("  [2] TRAIN LONGER  same thing, 50 epochs instead of the default")
        lines.append("      s.train(extra_epochs=50)")
        lines.append("")
        lines.append("  [3] PLAN ONLY  print what would happen, change nothing")
        lines.append("      s.plan('plan')")
        lines.append("")
        lines.append("  [4] INFER ONLY  nothing to load yet, will explain what is missing")
        lines.append("      s.infer('photo.jpg')")
    return "\n".join(lines)


def brief(
    session: Session,
    *,
    mode: str = "plan",
    extra_epochs: int | None = None,
    width: int | None = None,
) -> str:
    """Full pre-flight report for a session."""

    plan = session.plan(mode, extra_epochs)
    state = plan["state"]
    config = session.config

    out: list[str] = []
    out.append(rule("━"))
    out.append(" NIGHT FLASH — PRE-FLIGHT BRIEFING")
    out.append(rule("━"))
    out.append(kv("mode", f"{mode.upper()}   (plan / train / infer)"))
    out.append(kv("selected", f"arch={config.arch}  head={config.head}"))

    out.append(heading("1. WHAT THIS MODEL IS"))
    out.append(f"  {ARCH_DESCRIPTIONS[config.arch]}")
    out.append("")
    out.append(f"  Output head: {HEAD_DESCRIPTIONS[config.head]}")
    out.append("")
    out.append(architecture_table())
    out.append("")
    out.append(head_table())

    out.append(heading("2. CAPABILITIES  (each independently switchable)"))
    out.extend(_capability_lines(config))
    out.append("")
    out.extend(
        wrap(
            "The three extras are photometric 3-channel terms applied to the RGB residual. "
            "Each is gated by a learnable scalar initialised to zero, so a fresh model is "
            "exactly the plain residual and the extras earn their influence during training."
        )
    )

    out.append(heading("3. WHAT WILL BE TRAINED"))
    out.append(kv("parameters", f"{_param_count(session):,}"))
    out.append(kv("input", f"RGB{f' + person mask' if session.mask_dir else ' (no mask)'}"))
    out.append(kv("resolution", f"{session.image_size} x {session.image_size} (square crops)"))
    out.append(kv("batch size", session.batch_size))
    out.append(kv("device", session.device))
    out.append(kv("mixed precision", yes_no(session.use_amp)))
    out.append(kv("seed", session.seed))
    out.append(kv("optimiser", "AdamW, G lr 2e-4 / D lr 1e-4, betas (0.5, 0.999)"))
    out.append(kv("discriminator", "PatchGAN, 3 stride-2 stages"))
    out.append("")
    out.extend(
        wrap(
            "Unpaired training. The discriminator sees targets as targets; every other loss "
            "is a leash holding the output to the source. No paired images required."
        )
    )

    out.append(heading("4. WHERE EVERYTHING GOES"))
    out.append(_paths_block(session))
    out.append("")
    out.extend(
        wrap(
            "Each architecture and head gets its own folder, so switching never overwrites "
            "another run. Old checkpoints stay where they are."
        )
    )

    out.append(heading("5. WHAT WILL HAPPEN"))
    out.extend(_flow_lines(session, plan))

    out.append(heading("6. CURRENT STATE"))
    out.append(_state_block(session))

    out.append(heading("7. DECISION"))
    out.append(_decision_block(session, plan))
    out.append("")
    out.append(rule())
    return "\n".join(out)


def _param_count(session: Session) -> int:
    from .arch import build_generator

    try:
        return sum(p.numel() for p in build_generator(session.config).parameters())
    except Exception:
        return 0


def brief_and_print(session: Session, *, mode: str = "plan",
                    extra_epochs: int | None = None) -> str:
    text = brief(session, mode=mode, extra_epochs=extra_epochs)
    print(text)
    return text
