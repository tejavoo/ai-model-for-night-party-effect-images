"""``Session`` — one object that owns a run.

The goal is that a notebook never has to remember *where* it is. ``Session``
discovers what already exists on disk, derives the correct absolute epoch target,
and offers three explicit modes:

``plan``
    Print what *would* happen. Writes nothing, trains nothing.
``train``
    Resume if a checkpoint exists, otherwise start clean. Never silently no-ops.
``infer``
    Skip training entirely and just run the model.

Run directories are isolated per architecture and head, so switching
architectures never overwrites another run::

    runs/edgehat_freqsplit_night/
      config.json  checkpoints/  previews/  inferences/
      evaluations/ logs/  warm_start_report.json
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .arch import ARCHITECTURES, HEADS, build_generator, new_arch_config, run_generator
from .augment import AugmentConfig
from .config import ModelConfig
from .data import UnpairedImageDataset, make_dataloaders, make_mode_dataloaders
from .degrade import PROFILES, TRAIN_MODES, DegradeConfig, make_config
from .effects import apply_stochastic_effects
from .geometry import prepare, restore
from .losses import (
    LOSS_VERSION,
    adversarial_loss,
    content_loss,
    effect_regularization,
    identity_loss,
    moment_loss,
    paired_reconstruction,
    psnr,
)
from .models import PatchDiscriminator
from .progress import create_reporter
from .utils import ensure_dir, resolve_device, save_json, save_preview, seed_everything
from .validate import (
    ValidationMetrics,
    _metric_key,
    append_history,
    best_epoch,
    combine_metrics,
    evaluate_batch,
    render_strip,
)

Mode = Literal["plan", "train", "infer"]

SUBDIRS = ("checkpoints", "previews", "inferences", "evaluations", "logs")

AUGMENT_PRESETS = ("default", "conservative", "strong", "none")


def _augment_from_preset(preset: str) -> AugmentConfig:
    if preset == "conservative":
        return AugmentConfig.conservative()
    if preset == "strong":
        return AugmentConfig.strong()
    if preset == "none":
        return AugmentConfig(
            scale_min=1.0, scale_max=1.0, rotate_degrees=0.0, hflip=0.0, vflip=0.0
        )
    if preset != "default":
        raise ValueError(f"augment_preset must be one of {AUGMENT_PRESETS}, got {preset!r}")
    return AugmentConfig()


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-").lower()


@dataclass
class RunState:
    """What exists on disk for one run directory."""

    run_dir: Path
    exists: bool
    has_config: bool
    has_checkpoint: bool
    has_warm_start: bool
    last_epoch: int | None = None
    global_step: int | None = None
    saved_arch: str | None = None
    saved_head: str | None = None
    capabilities: list[str] = field(default_factory=list)
    checkpoint_bytes: int = 0
    preview_count: int = 0
    inference_count: int = 0
    issue: str | None = None
    saved_loss_version: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.issue:
            return f"BLOCKED: {self.issue}"
        if not self.exists:
            return "NEW - no run directory yet, will start from scratch"
        if not self.has_checkpoint:
            return "EMPTY - directory exists but no checkpoint, will start from scratch"
        return (
            f"RESUMABLE - epoch {self.last_epoch} of {self.last_epoch + 1} done, "
            f"global step {self.global_step}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "exists": self.exists,
            "status": self.status,
            "has_config": self.has_config,
            "has_checkpoint": self.has_checkpoint,
            "has_warm_start": self.has_warm_start,
            "last_epoch": self.last_epoch,
            "global_step": self.global_step,
            "saved_arch": self.saved_arch,
            "saved_head": self.saved_head,
            "capabilities": self.capabilities,
            "checkpoint_mb": round(self.checkpoint_bytes / 1e6, 1),
            "previews": self.preview_count,
            "inferences": self.inference_count,
            "issue": self.issue,
        }


class Session:
    """Owns one run: config, directories, state discovery, and execution."""

    def __init__(
        self,
        *,
        arch: str = "edgehat",
        head: str = "freqsplit",
        name: str = "night",
        source_dir: str | Path | None = None,
        target_dir: str | Path | None = None,
        val_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        runs_root: str | Path = "runs",
        image_size: int = 256,
        batch_size: int = 8,
        epochs: int = 50,
        device: str = "auto",
        use_amp: bool = True,
        seed: int = 0,
        num_workers: int = 2,
        preview_every: int = 250,
        log_every: int = 25,
        effects_strength: float = 0.5,
        halation: bool = True,
        subject_gate: bool = True,
        region_exposure: bool = True,
        resume: bool = True,
        augment: "AugmentConfig | None" = None,
        augment_preset: str = "default",
        snapshot_every: int = 0,
        snapshot_dataset: str | None = None,
        train_mode: str | None = None,
        degrade: "DegradeConfig | None" = None,
        degrade_profile: str = "moderate",
        paired_weight: float = 1.0,
        val_every: int = 1,
        val_count: int = 8,
        val_select: str = "fixed",
        val_cols: int = 3,
        val_split: float = 0.15,
        val_dir_override: str | Path | None = None,
        val_seed: int = 0,
        val_layout: str = "rows",
        keep_best: bool = True,
        keep_last: int = 3,
        ema_decay: float = 0.0,
        **arch_overrides: Any,
    ) -> None:
        if arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {ARCHITECTURES}, got {arch!r}")
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        if val_select not in ("fixed", "random"):
            raise ValueError(f"val_select must be 'fixed' or 'random', got {val_select!r}")
        if val_layout not in ("rows", "cols"):
            raise ValueError(f"val_layout must be 'rows' or 'cols', got {val_layout!r}")
        if train_mode is not None and train_mode not in TRAIN_MODES:
            raise ValueError(f"train_mode must be one of {TRAIN_MODES}, got {train_mode!r}")

        self.config: ModelConfig = new_arch_config(
            arch,
            head=head,
            use_halation=halation,
            use_subject_gate=subject_gate,
            use_region_exposure=region_exposure,
            **arch_overrides,
        )
        self.name = name
        self.source_dir = Path(source_dir) if source_dir else None
        self.target_dir = Path(target_dir) if target_dir else None
        self.val_dir = Path(val_dir) if val_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.use_amp = use_amp
        self.seed = seed
        self.num_workers = num_workers
        self.preview_every = preview_every
        self.log_every = log_every
        self.effects_strength = effects_strength
        self.resume_enabled = resume
        self.augment = augment or _augment_from_preset(augment_preset)
        self.snapshot_every = int(snapshot_every)
        self.snapshot_dataset = snapshot_dataset
        self.paired_weight = float(paired_weight)
        self.train_mode = train_mode
        self.degrade = make_config(degrade_profile) if degrade is None else degrade
        self.degrade.curriculum_epochs = int(epochs)
        self.keep_best = bool(keep_best)
        self.keep_last = max(1, int(keep_last))
        self.ema_decay = float(ema_decay)
        self._ema_state: dict[str, torch.Tensor] | None = None

        # Paired and sameset derive the source from the target pool, so a
        # separate source directory is meaningless for them. Rather than break
        # every existing two-pool script, an *unspecified* mode is resolved from
        # the arguments actually given: paired for a single pool, unpaired for
        # two. An explicitly requested mode is always honoured, and rejected
        # loudly if it cannot work.
        same_pool = bool(target_dir) and (
            source_dir is None or Path(source_dir).resolve() == Path(target_dir).resolve()
        )
        if train_mode is None:
            self.train_mode = "paired" if same_pool else "unpaired"
            if self.train_mode == "unpaired" and source_dir and target_dir:
                print(
                    f"note: source_dir and target_dir are different pools, so "
                    f"train_mode resolved to 'unpaired'. To build the source by "
                    f"degrading the target pool instead, pass only target_dir "
                    f"(or train_mode='paired' with matching directories)."
                )
        else:
            self.train_mode = train_mode
            if train_mode in ("paired", "sameset"):
                if not target_dir:
                    raise ValueError(
                        f"train_mode={train_mode!r} needs target_dir: it is the pool that "
                        "both the source and the target are drawn from."
                    )
                if not same_pool:
                    raise ValueError(
                        f"train_mode={train_mode!r} builds the source by degrading images "
                        f"from the target pool, so source_dir and target_dir must be the "
                        f"same directory. Got {source_dir!r} and {target_dir!r}. Either "
                        f"drop source_dir, or use train_mode='unpaired' to train across "
                        f"two pools."
                    )

        self.val_config = {
            "every": int(val_every),
            "count": int(val_count),
            "select": val_select,
            "cols": int(val_cols),
            "split": float(val_split),
            "dir": Path(val_dir_override) if val_dir_override else None,
            "seed": int(val_seed),
            "layout": val_layout,
        }

        folder = f"{arch}_{head}{'_' + name if name else ''}"
        self.run_dir = Path(runs_root) / folder
        for sub in SUBDIRS:
            ensure_dir(self.run_dir / sub)

    # ------------------------------------------------------------------ paths

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def latest(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def preview_dir(self) -> Path:
        return self.run_dir / "previews"

    @property
    def inference_dir(self) -> Path:
        return self.run_dir / "inferences"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "logs" / "train.log"

    @property
    def validation_dir(self) -> Path:
        return self.run_dir / "validations"

    # ------------------------------------------------------------- kaggle glue

    def auto_resume(self, *, required: bool = False) -> bool:
        """Pull in an attached Kaggle snapshot if one exists for this run.

        Call this at the top of every new Kaggle session instead of importing
        :func:`nightflash.kaggle.restore` by hand. Returns ``True`` when a
        snapshot was restored, ``False`` when you are starting fresh.

        Outside Kaggle it is a no-op that reports why, so the same notebook cell
        is safe in Colab.
        """

        from .kaggle import auto_restore, is_kaggle

        if not is_kaggle():
            print("not a Kaggle runtime; nothing to restore")
            return False
        restored = auto_restore(self.run_dir, required=required)
        if restored is not None:
            print(f"resumed {restored.name} at epoch {self.state().last_epoch + 1}")
            return True
        return False

    def snapshot(
        self, *, note: str | None = None, strict: bool = False
    ) -> "object | None":
        """Publish the run as a Kaggle Dataset version. No-op off Kaggle.

        Failures are reported and swallowed by default, because this is the
        call you make in the last cell before a session ends: a missing
        ``kaggle`` package or an expired token should print a warning, not lose
        the run you were trying to save. Pass ``strict=True`` to raise instead.
        """

        from .kaggle import is_kaggle, snapshot as _snapshot

        if not is_kaggle():
            print("not a Kaggle runtime; skip snapshot (use Colab Drive instead)")
            return None
        try:
            info = _snapshot(self.run_dir, dataset=self.snapshot_dataset, note=note)
        except Exception as exc:
            if strict:
                raise
            print(
                f"snapshot failed ({type(exc).__name__}: {exc}).\n"
                f"         The run is still on disk at {self.run_dir}. Save Version "
                "with output enabled, or download it before the session ends."
            )
            return None
        print(f"snapshot -> {info.dataset} v{info.version}")
        return info

    # ------------------------------------------------------------------ state

    def state(self) -> RunState:
        """Inspect the run directory without modifying anything."""

        state = RunState(
            run_dir=self.run_dir,
            exists=self.run_dir.exists(),
            has_config=(self.run_dir / "config.json").exists(),
            has_checkpoint=self.latest.exists(),
            has_warm_start=(self.run_dir / "warm_start_report.json").exists(),
        )
        if state.exists and state.has_config:
            saved = json.loads((self.run_dir / "config.json").read_text())
            model_config = ModelConfig.from_dict(saved.get("model_config", {}))
            state.saved_arch = model_config.arch
            state.saved_head = model_config.head
            state.capabilities = model_config.capability_names()
            if state.saved_arch != self.config.arch:
                state.issue = (
                    f"this run directory was created for arch={state.saved_arch!r} but you "
                    f"asked for {self.config.arch!r}; use a different name or runs_root"
                )
            elif state.saved_head != self.config.head:
                state.issue = (
                    f"this run directory was created for head={state.saved_head!r} but you "
                    f"asked for {self.config.head!r}"
                )

        if state.has_checkpoint:
            checkpoint = torch.load(self.latest, map_location="cpu", weights_only=False)
            state.last_epoch = int(checkpoint.get("epoch", -1))
            state.global_step = int(checkpoint.get("global_step", 0))
            state.checkpoint_bytes = self.latest.stat().st_size
            state.saved_loss_version = checkpoint.get("loss_version")
            saved_config = ModelConfig.from_dict(checkpoint.get("model_config", {}))
            if saved_config.arch != self.config.arch or saved_config.head != self.config.head:
                state.issue = (
                    f"checkpoint holds arch={saved_config.arch}/head={saved_config.head} but "
                    f"this session wants {self.config.arch}/{self.config.head}"
                )
            elif (
                state.saved_loss_version is not None
                and int(state.saved_loss_version) != LOSS_VERSION
            ):
                # Not fatal, but the user must know the two halves of the run
                # were optimised under different objectives.
                state.warnings.append(
                    f"checkpoint was written with loss_version={state.saved_loss_version}, "
                    f"this build uses {LOSS_VERSION}. The objective changed, so early "
                    f"epochs and later epochs are not directly comparable. Start a new "
                    f"run directory for a clean comparison."
                )

        state.preview_count = len(list(self.preview_dir.glob("*.png")))
        state.inference_count = len(list(self.inference_dir.glob("*")))
        return state

    def target_epoch(self, extra_epochs: int | None = None) -> int:
        """Absolute epoch to train to.

        Two different intents, and mixing them up is how a run silently trains
        forever:

        * ``extra_epochs`` given -> **relative**. ``train(extra_epochs=20)`` on a
          run at epoch 30 stops at epoch 50. This is the resume path, and it is
          why a resume can never land on an empty loop and train nothing.
        * ``extra_epochs`` omitted -> **absolute**. ``--epochs 50`` means "epoch
          50", so running it again at epoch 50 is a no-op rather than another
          50 epochs.
        """

        if extra_epochs is not None:
            state = self.state()
            start = 0 if state.last_epoch is None else state.last_epoch + 1
            return start + int(extra_epochs)
        return int(self.epochs)

    # ------------------------------------------------------------------ plan

    def plan(self, mode: Mode = "plan", extra_epochs: int | None = None) -> dict[str, Any]:
        """Describe exactly what would happen. Writes nothing."""

        state = self.state()
        goal = self.target_epoch(extra_epochs)
        start = 0 if state.last_epoch is None else state.last_epoch + 1
        would_train = mode == "train" and state.issue is None and goal > start

        return {
            "mode": mode,
            "run_dir": str(self.run_dir),
            "arch": self.config.arch,
            "head": self.config.head,
            "capabilities": self.config.capability_names(),
            "state": state.to_dict(),
            "start_epoch": start,
            "target_epoch": goal,
            "epochs_this_call": max(0, goal - start) if would_train else 0,
            "would_train": would_train,
            "action": self._action_text(mode, state, start, goal),
        }

    def _action_text(
        self, mode: Mode, state: RunState, start: int, goal: int
    ) -> str:
        if state.issue:
            return f"BLOCKED - {state.issue}"
        if mode == "infer":
            if not state.has_checkpoint:
                return "INFER - no checkpoint yet, nothing to run"
            return f"INFER - load {self.latest} and write to {self.inference_dir}, no training"
        if mode == "plan":
            if not state.has_checkpoint:
                return f"PLAN - would start fresh and train epochs 1..{goal}"
            return f"PLAN - would resume from epoch {start} and train to {goal}"
        if goal <= start:
            return (
                f"NO-OP GUARD - already at epoch {start - 1}; target {goal} would train "
                "zero epochs. Ask for more epochs."
            )
        if not state.has_checkpoint:
            return f"TRAIN - start fresh, epochs 1..{goal}"
        return f"TRAIN - resume from epoch {start}, train to {goal}"

    def write_config(self) -> None:
        save_json(
            self.run_dir / "config.json",
            {
                "model_config": self.config.to_dict(),
                "name": self.name,
                "image_size": self.image_size,
                "batch_size": self.batch_size,
                "device": self.device,
                "amp": self.use_amp,
                "seed": self.seed,
                "preview_every": self.preview_every,
                "log_every": self.log_every,
                "effects_strength": self.effects_strength,
                "source_dir": str(self.source_dir) if self.source_dir else None,
                "target_dir": str(self.target_dir) if self.target_dir else None,
                "val_dir": str(self.val_dir) if self.val_dir else None,
                "mask_dir": str(self.mask_dir) if self.mask_dir else None,
                # Bumping this makes older checkpoints visibly non-comparable
                # instead of silently mixing two different objectives.
                "loss_version": LOSS_VERSION,
                "train_mode": self.train_mode,
                "paired_weight": self.paired_weight,
            },
        )

    # ------------------------------------------------------------------ train

    def train(
        self,
        extra_epochs: int | None = None,
        *,
        mode: Mode = "train",
    ) -> dict[str, Any]:
        info = self.plan(mode=mode, extra_epochs=extra_epochs)
        if not info["would_train"]:
            return info
        if self.target_dir is None:
            raise ValueError("target_dir is required to train")
        if self.train_mode == "unpaired" and self.source_dir is None:
            raise ValueError("source_dir is required when train_mode='unpaired'")

        goal = int(info["target_epoch"])
        seed_everything(self.seed)
        self.write_config()
        device = resolve_device(self.device)
        use_mask = self.mask_dir is not None
        if use_mask:
            self.config.use_mask = True

        model = build_generator(self.config).to(device)
        discriminator = PatchDiscriminator().to(device)
        optimizer_g = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-4)

        start_epoch = int(info["start_epoch"])
        global_step = 0
        if self.resume_enabled and self.latest.exists():
            checkpoint = torch.load(self.latest, map_location=device, weights_only=False)
            saved = ModelConfig.from_dict(checkpoint.get("model_config", {}))
            if saved.arch == self.config.arch and saved.head == self.config.head:
                model.load_state_dict(checkpoint["model"])
                discriminator.load_state_dict(checkpoint["discriminator"])
                optimizer_g.load_state_dict(checkpoint["optimizer_g"])
                optimizer_d.load_state_dict(checkpoint["optimizer_d"])
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
                global_step = int(checkpoint.get("global_step", 0))
                saved_ema = checkpoint.get("ema")
                if saved_ema and self.ema_decay > 0.0:
                    self._ema_state = {k: torch.as_tensor(v) for k, v in saved_ema.items()}
                print(f"resumed from {self.latest} (epoch {start_epoch}, step {global_step})")

        train_loader, preview_loader, train_dataset, val_loader, val_dataset = make_mode_dataloaders(
            self.source_dir, self.target_dir,
            train_mode=self.train_mode,
            image_size=self.image_size, batch_size=self.batch_size,
            mask_dir=self.mask_dir, num_workers=self.num_workers, seed=self.seed,
            augment=self.augment, degrade=self.degrade,
            val_every=self.val_config["every"],
            val_count=self.val_config["count"],
            val_select=self.val_config["select"],
            val_split=self.val_config["split"],
            val_dir=self.val_config["dir"],
            val_seed=self.val_config["seed"],
        )
        preview_dataset = train_dataset
        if len(train_loader) == 0:
            raise ValueError("empty DataLoader; reduce --batch-size or add images")
        if self.train_mode != "unpaired":
            print(
                f"train_mode={self.train_mode}  pool={self.target_dir}  "
                f"train={len(train_dataset)}  "
                f"val={len(val_dataset) if val_dataset is not None else 0}  "
                f"val images are held out and never trained on"
            )

        amp_enabled = bool(self.use_amp and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
        reporter = create_reporter(mode="auto", log_every=self.log_every)

        print(f"arch={self.config.arch} head={self.config.head} "
              f"caps={self.config.capability_names() or '-'}")
        print(f"parameters: generator {sum(p.numel() for p in model.parameters()):,}")
        print(f"epochs {start_epoch}..{goal - 1}  device={device}")

        for epoch in range(start_epoch, goal):
            model.train()
            discriminator.train()
            # Varies the degradation draw every epoch, and drives the curriculum.
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
            if hasattr(preview_dataset, "set_epoch"):
                preview_dataset.set_epoch(epoch)
            reporter.start_epoch(epoch + 1, goal, len(train_loader))
            epoch_start = time.time()
            step_in_epoch = 0
            running: dict[str, float] = {}

            for batch in train_loader:
                source = batch["source"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                model_input = torch.cat([source, mask], dim=1) if use_mask else source

                optimizer_d.zero_grad(set_to_none=True)
                residual, effect_params, _ = run_generator(model, model_input, mask)
                styled = self._compose(source, residual, effect_params, training=True)
                loss_d = 0.5 * (
                    adversarial_loss(discriminator(target), True)
                    + adversarial_loss(discriminator(styled.detach()), False)
                )
                loss_d.backward()
                optimizer_d.step()

                optimizer_g.zero_grad(set_to_none=True)
                residual, effect_params, _ = run_generator(model, model_input, mask)
                deterministic = torch.clamp(source + residual, 0.0, 1.0)
                styled = self._compose(source, residual, effect_params, training=True)
                loss_adv = adversarial_loss(discriminator(styled), True)
                # L1 half on the pre-effect image so stochastic grain is not
                # penalised; edge half on the styled image so geometry is
                # verified end to end. See losses.content_loss.
                loss_content = content_loss(
                    styled, source, mask if use_mask else None,
                    texture_reference=deterministic,
                )
                loss_moment = moment_loss(styled, target)
                target_input = torch.cat([target, torch.zeros_like(mask)], dim=1) if use_mask else target
                t_residual, _, _ = run_generator(model, target_input, mask)
                loss_identity = identity_loss(torch.clamp(target + t_residual, 0.0, 1.0), target)
                loss_g = (
                    loss_adv + 1.5 * loss_content + 0.35 * loss_moment
                    + 0.75 * loss_identity + 0.02 * effect_regularization(effect_params)
                )
                # Direct supervision, only for the images in this batch that
                # actually have a corresponding target. In sameset most items
                # do not, so the term is applied per-item rather than per-batch.
                loss_paired = torch.zeros((), device=device)
                if self.paired_weight > 0.0 and "is_paired" in batch:
                    flags = batch["is_paired"].to(device).view(-1, 1, 1, 1)
                    per_item = paired_reconstruction(
                        deterministic, target, mask if use_mask else None, reduction="none"
                    )
                    active = flags.sum()
                    if float(active.detach().cpu()) > 0.0:
                        loss_paired = (per_item * flags).sum() / active
                        loss_g = loss_g + self.paired_weight * loss_paired
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_g.step()

                if self.ema_decay > 0.0:
                    self._ema_update(model)

                global_step += 1
                step_in_epoch += 1
                metrics = {
                    "loss_d": float(loss_d.detach().cpu()),
                    "loss_g": float(loss_g.detach().cpu()),
                    "adv": float(loss_adv.detach().cpu()),
                    "content": float(loss_content.detach().cpu()),
                    "identity": float(loss_identity.detach().cpu()),
                }
                if float(loss_paired.detach().cpu()) > 0.0:
                    metrics["paired"] = float(loss_paired.detach().cpu())
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value
                reporter.update(step_in_epoch, metrics)

                if self.preview_every > 0 and global_step % self.preview_every == 0:
                    self._preview(model, preview_loader, device, epoch, global_step, use_mask)
                    self._save(model, discriminator, optimizer_g, optimizer_d, epoch, global_step)
                    print(f"preview + checkpoint at step {global_step}")

            reporter.finish_epoch(
                epoch + 1, goal, time.time() - epoch_start,
                {k: v / max(step_in_epoch, 1) for k, v in running.items()},
            )
            self._save(model, discriminator, optimizer_g, optimizer_d, epoch, global_step)
            self._preview(model, preview_loader, device, epoch, global_step, use_mask, name=f"epoch-{epoch+1:04d}")

            if val_loader is not None and self.val_config["every"] > 0 \
                    and (epoch + 1) % self.val_config["every"] == 0:
                val_metrics = self._validate(model, val_loader, val_dataset, device, epoch, use_mask)
                if val_metrics is not None:
                    # Keep-best must read the history *before* this epoch is
                    # appended, or it compares the current score against itself
                    # and never writes on the first epoch.
                    self._maybe_keep_best(model, discriminator, optimizer_g,
                                          optimizer_d, epoch, global_step, val_metrics)
                    append_history(self.validation_dir / "history.csv",
                                   val_metrics.as_row(epoch + 1))

            if self.snapshot_every > 0 and (epoch + 1) % self.snapshot_every == 0:
                self._maybe_snapshot(epoch + 1)

        return {**info, "completed": True, "global_step": global_step}

    def _maybe_snapshot(self, epoch: int) -> None:
        """Publish the run as a Kaggle Dataset version, never failing the run.

        A snapshot is a convenience, not a correctness requirement, so any error
        is downgraded to a warning: losing network access mid-session must not
        destroy an hour of training.
        """

        from .kaggle import is_kaggle, snapshot

        if not is_kaggle():
            print("snapshot requested but this is not a Kaggle runtime; skipping")
            return
        try:
            info = snapshot(
                self.run_dir,
                dataset=self.snapshot_dataset,
                note=f"epoch {epoch}",
            )
            print(f"epoch {epoch}: snapshot -> {info.dataset} v{info.version}")
        except Exception as exc:
            print(f"epoch {epoch}: snapshot failed ({exc}); training continues")

    # ------------------------------------------------------------------ infer

    def load(self, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
        if not self.latest.exists():
            raise FileNotFoundError(
                f"no checkpoint at {self.latest}. Run s.train() first, or check s.state()."
            )
        checkpoint = torch.load(self.latest, map_location=device, weights_only=False)
        saved = ModelConfig.from_dict(checkpoint.get("model_config", {}))
        model = build_generator(saved).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, checkpoint

    def infer(
        self,
        inputs: str | Path | Sequence[str | Path],
        *,
        seeds: int = 1,
        save_comparison: bool = False,
        device: str | None = None,
        tile: int = 0,
        tile_overlap: int = 128,
        pad_mode: str = "edge",
        max_edge: int = 0,
    ) -> list[Path]:
        """Run inference only. Never touches the training loop.

        Parameters
        ----------
        tile:
            ``0`` (default) processes the whole frame: nothing is cropped or
            stretched, but the model sees the image downscaled to ``image_size``.
            A positive value processes overlapping tiles of that size.
        tile_overlap:
            Overlap in pixels. 25% of the tile is the usual starting point.
        pad_mode:
            What the model sees outside the frame for edge tiles.
        max_edge:
            Downscale the long edge to at most this many pixels, **preserving
            aspect ratio**, before any other processing. ``0`` disables it.
            Useful for batching or for a model trained at a small ``image_size``.

        ``tile`` should match the ``image_size`` the model was trained at. A
        mismatch is warned about rather than blocked, because running a large
        tile through a small-crop model is occasionally useful.
        """

        from PIL import Image, ImageOps

        from .tiles import apply_tiled, estimate_tiles, tile_count

        dev = resolve_device(device or self.device)
        model, _ = self.load(dev)
        paths = _expand_inputs(inputs)
        written: list[Path] = []
        fit = "letterbox"

        if tile and tile != self.image_size:
            print(
                f"warning: tile={tile} but this model was trained at image_size={self.image_size}.\n"
                f"         Matching them is the reliable choice: a small tile gives the\n"
                f"         network a smaller context than it trained on, a large tile puts\n"
                f"         its receptive field in a different regime."
            )
        if tile and self.config.use_region_exposure:
            print(
                "note: tiled inference makes region-exposure tile-local. "
                "Train without --region-exposure if you need global regions."
            )

        def run_frame(image, seed_index: int):
            """One letterboxed pass. Returns a PIL image of the same size."""

            array, transform = prepare(image, self.image_size, fit=fit)
            source = torch.from_numpy(array).unsqueeze(0).to(dev)
            residual, effect_params, _ = run_generator(model, source)
            deterministic = torch.clamp(source + residual, 0.0, 1.0)
            restored = restore(deterministic[0].cpu().numpy(), transform)
            rgb = torch.from_numpy(
                np.asarray(restored, dtype=np.float32).copy() / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(dev)
            styled = apply_stochastic_effects(
                rgb, effect_params, seed=self.seed + seed_index,
                strength=self.effects_strength,
            )
            return restore(styled[0].cpu().numpy(), transform)

        with torch.no_grad():
            for path in paths:
                image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
                original_size = image.size
                if max_edge and max(image.size) > max_edge:
                    scale = max_edge / max(image.size)
                    image = image.resize(
                        (max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                    print(
                        f"{path.name}: downscaled {original_size} -> {image.size} "
                        f"(long edge {max_edge}, aspect preserved)"
                    )

                for seed_index in range(seeds):
                    if tile:
                        count = tile_count(image.width, image.height, tile, tile_overlap)
                        info = estimate_tiles(image.width, image.height, tile, tile_overlap)
                        print(
                            f"{path.name} {image.size} -> tiled: {info['tiles']} tiles "
                            f"of {tile}px, {info['megapixels_processed']} MP processed"
                        )

                        def process(piece, seed_index=seed_index):
                            return run_frame(piece, seed_index)

                        out = apply_tiled(
                            image, process, tile_size=tile,
                            overlap=tile_overlap, pad_mode=pad_mode,
                        )
                    else:
                        out = run_frame(image, seed_index)

                    suffix = "" if seeds == 1 else f"-seed{seed_index}"
                    mode = "" if not tile else f"-t{tile}"
                    if max_edge:
                        mode += f"-m{max_edge}"
                    destination = self.inference_dir / f"{path.stem}{mode}{suffix}.png"
                    if save_comparison:
                        canvas = Image.new("RGB", (image.width * 2, image.height))
                        canvas.paste(image, (0, 0))
                        canvas.paste(out, (image.width, 0))
                        canvas.save(destination)
                    else:
                        out.save(destination)
                    written.append(destination)
                    print(f"{path.name} -> {destination.name}  {out.size}")
        return written

    # ----------------------------------------------------------------- helpers

    def _compose(
        self, source: Tensor, residual: Tensor, effect_params: Tensor, *, training: bool
    ) -> Tensor:
        deterministic = torch.clamp(source + residual, 0.0, 1.0)
        if training:
            return apply_stochastic_effects(
                deterministic, effect_params, seed=None, strength=self.effects_strength
            )
        return deterministic

    def _preview(
        self, model, preview_loader, device, epoch: int, global_step: int,
        use_mask: bool, name: str | None = None,
    ) -> None:
        model.eval()
        with torch.no_grad():
            batch = next(iter(preview_loader))
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            model_input = torch.cat([source, mask], dim=1) if use_mask else source
            residual, effect_params, _ = run_generator(model, model_input, mask)
            styled = apply_stochastic_effects(
                torch.clamp(source + residual, 0.0, 1.0), effect_params,
                seed=424242, strength=self.effects_strength,
            )
        filename = name or f"step-{global_step:07d}"
        if not filename.endswith(".png"):
            filename = f"{filename}.png"
        save_preview(self.preview_dir / filename, source[0], styled[0], target[0])
        model.train()

    def _validate(
        self, model, val_loader, val_dataset, device, epoch: int, use_mask: bool
    ) -> "ValidationMetrics | None":
        """Score the held-out set and write one strip image.

        The effects seed is pinned per epoch so two strips are comparable, and
        the degradation itself is already pinned by the dataset's filename-derived
        seeds. Without both, epoch N and epoch N+1 would be different problems.
        """

        if val_loader is None or val_dataset is None or len(val_dataset) == 0:
            return None

        model.eval()
        rows: list[dict[str, Any]] = []
        totals: list[ValidationMetrics] = []
        with torch.no_grad():
            for batch in val_loader:
                source = batch["source"].to(device)
                target = batch["target"].to(device)
                mask = batch["mask"].to(device)
                model_input = torch.cat([source, mask], dim=1) if use_mask else source
                residual, effect_params, _ = run_generator(model, model_input, mask)
                deterministic = torch.clamp(source + residual, 0.0, 1.0)
                styled = apply_stochastic_effects(
                    deterministic, effect_params,
                    seed=self.val_config["seed"] + epoch,
                    strength=self.effects_strength,
                )
                totals.append(evaluate_batch(batch, styled, deterministic))
                paths = batch.get("target_path", [])
                for i in range(source.shape[0]):
                    name = Path(paths[i]).name if i < len(paths) else f"{i}"
                    rows.append({
                        "index": len(rows) + 1,
                        "filename": name,
                        "source": source[i].cpu(),
                        "output": styled[i].cpu(),
                        "target": target[i].cpu(),
                    })
        model.train()

        if not rows:
            return None
        summary = combine_metrics(totals)

        strip = render_strip(
            rows,
            columns=self.val_config["cols"],
            layout=self.val_config["layout"],
            title=f"validation  epoch {epoch + 1}  {self.train_mode}  "
                  f"degrade={self.degrade.resolved(epoch).profile}",
            metrics=summary,
        )
        ensure_dir(self.validation_dir)
        strip.save(self.validation_dir / f"epoch-{epoch + 1:04d}.png")
        strip.save(self.validation_dir / "latest.png")
        print(
            f"validation epoch {epoch + 1}: n={summary.images}  "
            f"PSNR {summary.psnr:.2f}dB  SSIM {summary.ssim:.4f}  L1 {summary.l1:.4f}  "
            f"-> {self.validation_dir}/epoch-{epoch + 1:04d}.png"
        )
        return summary

    def _maybe_keep_best(
        self, model, discriminator, optimizer_g, optimizer_d, epoch, global_step,
        metrics: "ValidationMetrics",
    ) -> None:
        """Write ``best.pt`` when validation PSNR improves.

        Only meaningful in ``paired`` mode. In ``sameset`` there is no
        correspondence between prediction and target, so PSNR is measuring
        nothing and ranking on it would be meaningless.
        """

        if not self.keep_best or self.train_mode != "paired":
            return
        previous = best_epoch(self.validation_dir / "history.csv")
        score = _metric_key(metrics)
        if previous is not None and score <= previous[1]:
            return
        self._write_checkpoint(
            self.checkpoint_dir / "best.pt", model, discriminator,
            optimizer_g, optimizer_d, epoch, global_step,
        )
        print(f"new best validation PSNR {score:.2f}dB -> {self.checkpoint_dir}/best.pt")

    def _ema_update(self, model) -> None:
        if self._ema_state is None:
            self._ema_state = {k: v.detach().clone().float()
                               for k, v in model.state_dict().items()}
            return
        decay = self.ema_decay
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if key in self._ema_state and value.dtype.is_floating_point:
                    self._ema_state[key].mul_(decay).add_(value.detach().float(), alpha=1 - decay)
                elif key in self._ema_state:
                    self._ema_state[key] = value.detach().clone()

    def _write_checkpoint(self, path, model, discriminator, optimizer_g, optimizer_d,
                          epoch, global_step) -> None:
        ensure_dir(Path(path).parent)
        torch.save({
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "model_config": self.config.to_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "loss_version": LOSS_VERSION,
            "train_mode": self.train_mode,
            "ema": self._ema_state,
        }, path)

    def _save(self, model, discriminator, optimizer_g, optimizer_d, epoch, global_step) -> None:
        self._write_checkpoint(self.latest, model, discriminator, optimizer_g,
                               optimizer_d, epoch, global_step)
        self._write_checkpoint(
            self.checkpoint_dir / f"epoch-{epoch + 1:04d}.pt", model, discriminator,
            optimizer_g, optimizer_d, epoch, global_step,
        )
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        """Keep the newest ``keep_last`` epoch files; ``best.pt`` is never touched.

        60 epochs of a 17 MB checkpoint is a gigabyte, which fills Kaggle's disk
        and makes every snapshot slow.
        """

        if self.keep_last <= 0:
            return
        files = sorted(self.checkpoint_dir.glob("epoch-*.pt"))
        for stale in files[: max(0, len(files) - self.keep_last)]:
            try:
                stale.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------- misc

    def reset(self, *, archive: bool = True, keep_config: bool = True) -> Path | None:
        """Clear checkpoints and outputs so the next run starts from zero.

        By default the old run is moved to ``<run_dir>_archive_<n>`` rather than
        deleted, so a 40-epoch baseline is never lost to a stray call. Pass
        ``archive=False`` to delete outright.
        """

        if not self.run_dir.exists():
            return None

        destination: Path | None = None
        if archive and any(self.run_dir.iterdir()):
            index = 1
            while (candidate := self.run_dir.with_name(f"{self.run_dir.name}_archive_{index}")).exists():
                index += 1
            shutil.move(str(self.run_dir), str(candidate))
            destination = candidate
        else:
            # In-place wipe: the subdirectories must actually be removed.
            for sub in ("checkpoints", "previews", "inferences", "evaluations", "logs"):
                target = self.run_dir / sub
                if target.exists():
                    shutil.rmtree(target)

        for sub in SUBDIRS:
            ensure_dir(self.run_dir / sub)
        if not keep_config and (self.run_dir / "config.json").exists():
            (self.run_dir / "config.json").unlink()
        if destination is not None:
            print(f"archived previous run -> {destination}")
        print(f"reset {self.run_dir} (checkpoints and previews cleared)")
        return destination

    def __repr__(self) -> str:
        return f"Session(arch={self.config.arch!r}, head={self.config.head!r}, run_dir={str(self.run_dir)!r})"


def _expand_inputs(inputs: str | Path | Sequence[str | Path]) -> list[Path]:
    from .data import IMAGE_EXTENSIONS, list_images

    if isinstance(inputs, (str, Path)):
        path = Path(inputs).expanduser()
        if path.is_dir():
            return list_images(path)
        if path.is_file():
            return [path]
        raise FileNotFoundError(path)
    return [Path(p).expanduser() for p in inputs]
