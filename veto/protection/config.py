import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from veto.configs.project_config import ProjectConfig
from veto.protection.wrappers.base import ProtectionModel


@dataclass
class DoubleStreamConfig:
    """VETO hooks on ``transformer_blocks`` (double-stream cross-attention)."""

    enabled: bool = True
    layer_indices: list[int] = field(default_factory=lambda: [0])
    entropy_slices: list[str] = field(
        default_factory=lambda: [
            "canvas_text",
            "reference_text",
            "text_canvas",
            "text_reference",
        ]
    )


@dataclass
class SingleStreamConfig:
    """VETO hooks on ``single_transformer_blocks`` (joint [text | canvas | reference])."""

    enabled: bool = True
    layer_indices: list[int] = field(default_factory=lambda: [0])
    entropy_slices: list[str] = field(default_factory=lambda: ["canvas_reference"])


@dataclass
class ProtectionConfig:
    project: ProjectConfig
    dataset_dir: Path
    model: ProtectionModel = ProtectionModel.FLUX2
    surrogate_prompt: str = ""
    inference_steps: int = 10
    num_timesteps_per_step: int = 3
    guidance_scale: float = 4.0
    fibo_max_sequence_length: int = 3000
    fibo_base_json: dict | None = None
    double_stream: DoubleStreamConfig = field(default_factory=DoubleStreamConfig)
    single_stream: SingleStreamConfig = field(default_factory=SingleStreamConfig)
    entropy_eps: float = 1e-8
    epsilon: int = 8
    step_size: int = 2
    steps: int = 10
    momentum_decay: float = 0.0
    weight_decay: float = 0.0
    constraint_type: str = "default_epsilon"
    constraint_threshold: float = None
    wandb_enabled: bool = False
    wandb_project: str = "veto"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    wandb_log_every_n_steps: int = 1
    run_name_template: str = (
        "veto_<model>_eps=<epsilon>_steps=<steps>_<constraint_type>_<timestamp>"
    )
    device: str = "cuda"
    seed: int = 0
    image_size: int = 1024


def materialize_run_name(config: ProtectionConfig) -> str:
    """Build evaluation ``run_id`` from ``run_name_template`` only."""
    tpl = (config.run_name_template or "").strip()
    if not tpl:
        return "protection"

    def subst(expr: str):
        if expr == "timestamp":
            return datetime.now().strftime("%Y%m%d_%H%M%S")
        if expr in ("model", "model_family"):
            return config.model.value
        if hasattr(config, expr):
            v = getattr(config, expr)
            if v is None:
                return ""
            return str(v)
        return None

    def replace_placeholder(match: re.Match[str]) -> str:
        expr = match.group(1)
        v = subst(expr)
        if v is None:
            return match.group(0)
        return v

    return re.sub(r"<([^>]+)>", replace_placeholder, tpl)
