import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from veto.configs.project_config import ProjectConfig
from veto.protection.attention.entropy import ALL_ENTROPY_SLICES, VALID_ENTROPY_SLICES
from veto.protection.config import (
    DoubleStreamConfig,
    ProtectionConfig,
    SingleStreamConfig,
)
from veto.protection.wrappers.base import ProtectionModel
from veto.protection.wrappers.factory import resolve_model

_DEPRECATED_STREAM_KEYS = frozenset({"layers_mode", "last_n_layers"})


def _parse_stream_block(
    cfg: dict[str, Any],
    *,
    path: str,
) -> tuple[bool, list[str], list[int] | None]:
    """Parse ``protection.<stream>`` (``enabled``, ``layer_indices``, ``entropy_slices``)."""
    if _DEPRECATED_STREAM_KEYS & cfg.keys():
        raise ValueError(
            f"{path} no longer supports layers_mode or last_n_layers; use layer_indices only"
        )

    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return False, [], None

    entropy_slices = [str(s).strip() for s in (cfg.get("entropy_slices") or [])]
    if len(entropy_slices) == 1 and entropy_slices[0].lower() == "all":
        entropy_slices = list(ALL_ENTROPY_SLICES)

    if not entropy_slices:
        raise ValueError(
            f"{path} is enabled but entropy_slices is empty; "
            f"set enabled: false to disable the stream, or list entropy slices"
        )

    unknown = set(entropy_slices) - VALID_ENTROPY_SLICES
    if unknown:
        raise ValueError(
            f"{path}.entropy_slices contains unknown values {sorted(unknown)}; "
            f"use: {sorted(VALID_ENTROPY_SLICES)}"
        )

    raw_indices = cfg.get("layer_indices")
    if raw_indices is None:
        raise ValueError(
            f"{path}.layer_indices is required when the stream is enabled"
        )
    layer_indices = [int(i) for i in raw_indices]
    if not layer_indices:
        raise ValueError(f"{path}.layer_indices must be non-empty")
    return True, entropy_slices, layer_indices


def save_config_snapshot(
    source_path: Path | str,
    dest_path: Path | str,
    *,
    run_id: str,
) -> Path:
    """Save the launch YAML into the run output tree with a frozen ``run_name_template``."""
    src = Path(source_path)
    dst = Path(dest_path)
    if not src.is_file():
        raise FileNotFoundError(f"Config not found: {src}")

    text = src.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'^(\s*run_name_template:\s*).*$',
        rf'\1"{run_id}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise ValueError(
            f"Could not freeze run_name_template in config snapshot: {src}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new_text, encoding="utf-8")
    return dst


def load_protection_config(config_path: str) -> ProtectionConfig:
    raw = OmegaConf.load(config_path)
    d: dict[str, Any] = OmegaConf.to_container(raw, resolve=True)

    project = ProjectConfig()

    data = d.get("data") or {}
    if not str(data.get("dataset_dir", "")).strip():
        raise ValueError("data.dataset_dir is required")
    dataset_dir = Path(data.get("dataset_dir", "")).expanduser()

    protection_cfg = d.get("protection") or {}
    model = resolve_model(protection_cfg)
    if model not in (ProtectionModel.FLUX2, ProtectionModel.FIBO_EDIT):
        raise ValueError(
            f"protection.model must be flux2 or fibo_edit, got {model.value!r}"
        )

    surrogate_prompt = str(protection_cfg.get("surrogate_prompt", ""))
    inference_steps = int(protection_cfg.get("inference_steps", 10))
    num_timesteps_per_step = int(protection_cfg.get("num_timesteps_per_step", 3))
    guidance_scale = float(protection_cfg.get("guidance_scale", 4.0))
    entropy_eps = float(protection_cfg.get("entropy_eps", 1e-8))

    fibo_max_sequence_length = int(protection_cfg.get("fibo_max_sequence_length", 3000))
    if fibo_max_sequence_length <= 0 or fibo_max_sequence_length > 3000:
        raise ValueError("protection.fibo_max_sequence_length must be in (0, 3000]")

    fibo_base_json_raw = protection_cfg.get("fibo_base_json")
    fibo_base_json = None
    if fibo_base_json_raw is not None:
        if not isinstance(fibo_base_json_raw, dict):
            raise ValueError("protection.fibo_base_json must be a mapping when set")
        fibo_base_json = dict(fibo_base_json_raw)

    if model == ProtectionModel.FIBO_EDIT and guidance_scale < 1.0:
        raise ValueError("protection.guidance_scale must be >= 1.0 for fibo_edit")

    ds_cfg = protection_cfg.get("double_stream")
    if ds_cfg is None:
        raise ValueError("protection.double_stream is required")
    if not isinstance(ds_cfg, dict):
        raise ValueError("protection.double_stream must be a mapping")

    ss_cfg = protection_cfg.get("single_stream")
    if ss_cfg is None:
        raise ValueError("protection.single_stream is required")
    if not isinstance(ss_cfg, dict):
        raise ValueError("protection.single_stream must be a mapping")

    ds_enabled, ds_slices, ds_indices = _parse_stream_block(
        ds_cfg,
        path="protection.double_stream",
    )
    ss_enabled, ss_slices, ss_indices = _parse_stream_block(
        ss_cfg,
        path="protection.single_stream",
    )

    if not ds_enabled and not ss_enabled:
        raise ValueError(
            "At least one of protection.double_stream or protection.single_stream "
            "must have enabled: true"
        )
    if not ds_slices and not ss_slices:
        raise ValueError(
            "At least one entropy slice must be configured on an enabled stream"
        )

    if inference_steps <= 0:
        raise ValueError("protection.inference_steps must be positive")
    if num_timesteps_per_step <= 0:
        raise ValueError("protection.num_timesteps_per_step must be positive")
    if num_timesteps_per_step > inference_steps:
        raise ValueError(
            "protection.num_timesteps_per_step must be <= inference_steps"
        )

    pert_cfg = d.get("perturbation") or {}
    run = d.get("run") or {}
    epsilon = int(pert_cfg.get("epsilon", 8))
    step_size = int(pert_cfg.get("step_size", 2))
    steps = int(pert_cfg.get("steps", 10))
    momentum_decay = float(pert_cfg.get("momentum_decay", 0.0))
    if momentum_decay < 0:
        raise ValueError("perturbation.momentum_decay must be >= 0")
    weight_decay = float(pert_cfg.get("weight_decay", 0.0))
    if weight_decay < 0:
        raise ValueError("perturbation.weight_decay must be >= 0")

    constraint_cfg = pert_cfg.get("constraint") or {}
    constraint_type = str(constraint_cfg.get("type", "default_epsilon"))
    threshold = constraint_cfg.get("threshold")
    constraint_threshold = float(threshold) if threshold is not None else None


    wandb_cfg = run.get("wandb") or {}
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    wandb_project = str(wandb_cfg.get("project", "veto"))
    wandb_entity = str(wandb_cfg.get("entity", "")).strip()

    if wandb_enabled and not wandb_entity:
        raise ValueError(
            "When run.wandb.enabled is true, set run.wandb.entity in the config"
        )

    wandb_mode = str(wandb_cfg.get("mode", "online"))
    wandb_log_every_n_steps = int(wandb_cfg.get("log_every_n_steps", 1))

    if epsilon <= 0 or step_size <= 0 or steps <= 0:
        raise ValueError("perturbation.epsilon, step_size, steps must be positive")
    if wandb_log_every_n_steps <= 0:
        raise ValueError("run.wandb.log_every_n_steps must be positive")

    run_name_template = run.get(
        "run_name_template",
        "veto_eps=<epsilon>_steps=<steps>_<constraint_type>_<timestamp>",
    )

    return ProtectionConfig(
        project=project,
        dataset_dir=dataset_dir.resolve(),
        model=model,
        surrogate_prompt=surrogate_prompt,
        inference_steps=inference_steps,
        num_timesteps_per_step=num_timesteps_per_step,
        guidance_scale=guidance_scale,
        fibo_max_sequence_length=fibo_max_sequence_length,
        fibo_base_json=fibo_base_json,
        double_stream=DoubleStreamConfig(
            enabled=ds_enabled,
            layer_indices=ds_indices or [],
            entropy_slices=ds_slices,
        ),
        single_stream=SingleStreamConfig(
            enabled=ss_enabled,
            layer_indices=ss_indices or [],
            entropy_slices=ss_slices,
        ),
        entropy_eps=entropy_eps,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        momentum_decay=momentum_decay,
        weight_decay=weight_decay,
        constraint_threshold=constraint_threshold,
        constraint_type=constraint_type,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        wandb_log_every_n_steps=wandb_log_every_n_steps,
        run_name_template=str(run_name_template),
        device=run.get("device", "cuda"),
        seed=int(run.get("seed", 0)),
        image_size=int(run.get("image_size", 512)),
    )
