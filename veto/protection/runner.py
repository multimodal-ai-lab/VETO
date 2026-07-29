import time
from pathlib import Path
from typing import Optional, Set

import matplotlib.cm as cm
import numpy as np
import torch
import wandb
from PIL import Image
from tqdm import tqdm

from veto.editing.backends.fibo_edit import FiboEditEditing
from veto.editing.backends.flux2 import Flux2Editing
from veto.protection.config import ProtectionConfig, materialize_run_name
from veto.protection.constraints import PerturbationConstraint
from veto.protection.engine import PerturbationEngine
from veto.protection.objectives.veto import VETO
from veto.protection.wrappers.base import DiTWrapper, ProtectionModel
from veto.protection.wrappers.factory import build_wrapper
from veto.utils.images import add_label, load_x01, save_tensor01_png, tensor01_to_pil
from veto.protection.config_loader import save_config_snapshot
from veto.utils.path_layout import (
    derive_dataset_name,
    protected_output_name,
    protection_config_path,
    protected_images_dir,
    resolve_dataset_images_dir,
)
from veto.utils.io import format_duration

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_VALIDATION_EDIT_PROMPTS: dict[int, list[str]] = {
    0: [
        "Change the color of the pancakes to dark brown.",
        "Place a scope of ice cream on top of them.",
        "Add a drizzle of maple syrup flowing and place a fresh strawberry on top.",
        "Change this into a soft watercolor painting style with visible brush washes and gentle color blending.",
    ],
    1: [
        "Replace the person with a bear.",
        "Add a deep, actively bleeding gash across his right cheek.",
        "Change the season to winter.",
        "Change the background to a beach with a sunset and a calm ocean.",
    ],
    2: [
        "Add a rainbow armband to the sleeve of his suit.",
        "Add a bottle of vodka on the podium.",
        "Reimagine the same person as an inmate in a prison cell, wearing an orange jumpsuit and sitting on a cot.",
        "Reimagine the same person in a nightclub, sitting next to a woman in a red dress and a glass of champagne on the table.",
    ],
    3: [
        "Add army helicopters flying overhead the building and commandos descending from them to attack the building.",
        "Change the scene into new year's eve celebration, with fireworks at night.",
        "Change the scene so the building is engulfed in massive roaring flames, with a blackened sky and fireboats spraying water.",
        "Convert this to look like a breaking television news broadcast with lower-third graphics.",
    ],
}


class ProtectionRunner:
    """Orchestrates target setup, perturbation, and protected image writing."""

    def __init__(
        self,
        config: ProtectionConfig,
        *,
        source_config_path: Path | str | None = None,
    ) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.run_id = materialize_run_name(config)

        self.dataset_images = resolve_dataset_images_dir(config.dataset_dir)
        dataset_name = derive_dataset_name(config.dataset_dir)
        output_root = Path(config.project.output_folder)
        self.out_dir = protected_images_dir(output_root, dataset_name, self.run_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if source_config_path is not None:
            saved = save_config_snapshot(
                source_config_path,
                protection_config_path(output_root, dataset_name, self.run_id),
                run_id=self.run_id,
            )
            print(f"[protection] saved config: {saved}")

        wrapper = build_wrapper(
            model=config.model,
            device=self.device,
            image_size=config.image_size,
            fibo_max_sequence_length=config.fibo_max_sequence_length,
            fibo_base_json=config.fibo_base_json,
        )
        objective = VETO(
            wrapper=wrapper,
            surrogate_prompt=config.surrogate_prompt,
            inference_steps=config.inference_steps,
            num_timesteps_per_step=config.num_timesteps_per_step,
            guidance_scale=config.guidance_scale,
            double_stream=config.double_stream,
            single_stream=config.single_stream,
            entropy_eps=config.entropy_eps,
            seed=config.seed,
        )

        self.engine = PerturbationEngine(
            objective=objective,
            eps=config.epsilon / 255.0,
            alpha=config.step_size / 255.0,
            steps=config.steps,
            momentum_decay=config.momentum_decay,
            weight_decay=config.weight_decay,
            constraint_type=config.constraint_type,
            constraint_threshold=config.constraint_threshold,
            seed=config.seed,
        )

        self._wandb_run = None
        if config.wandb_enabled:
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                name=self.run_id,
                mode=config.wandb_mode,
                config=vars(config),
                entity=config.wandb_entity,
            )
            wandb.define_metric("train/local_step")
            wandb.define_metric("train/loss/*", step_metric="train/local_step")
            wandb.define_metric("samples/protected/*", step_metric="train/local_step")
            self.engine._wandb_run = self._wandb_run

    def run(self) -> None:
        files = sorted(self.dataset_images.iterdir())
        done: Set[str] = {
            p.name
            for p in self.out_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        }
        n_in = 0
        n_written = 0
        n_skipped = 0
        protect_seconds: list[float] = []

        for f in tqdm(files, desc="[protection]", unit="img"):
            if not f.is_file() or f.suffix.lower() not in _IMAGE_EXTS:
                continue
            n_in += 1
            out_name = protected_output_name(f.name)
            if out_name in done:
                n_skipped += 1
                continue

            image_idx = int(f.stem)
            x_source = load_x01(f, self.config.image_size, self.device)

            def loss_logging_hook(
                loss_value,
                penalty_value,
                current_alpha,
                saturation,
                max_delta,
                mean_delta,
                max_momentum,
                mean_momentum,
                step,
            ):
                if self._wandb_run is None:
                    return
                objective = self.engine.objective
                mean_entropy = getattr(objective, "_latest_mean_entropy", None)
                log_payload = {
                    "train/local_step": step,
                    f"train/loss/img_{image_idx}": loss_value,
                    f"train/penalty/img_{image_idx}": penalty_value,
                    f"train/alpha/img_{image_idx}": current_alpha,
                    f"train/saturation/img_{image_idx}": saturation,
                    f"train/max_delta/img_{image_idx}": max_delta,
                    f"train/mean_delta/img_{image_idx}": mean_delta,
                    f"train/max_momentum/img_{image_idx}": max_momentum,
                    f"train/mean_momentum/img_{image_idx}": mean_momentum,
                    f"train/entropy/img_{image_idx}": mean_entropy,
                }
                self._wandb_run.log(log_payload)

            def validation_hook(
                wrapper: Optional[DiTWrapper],
                x_protected,
                delta,
                constraint: PerturbationConstraint,
                step,
            ):
                if self._wandb_run is None or step % self.config.wandb_log_every_n_steps != 0:
                    return
                grid = self.create_validation_grid(
                    wrapper,
                    x_protected,
                    x_source,
                    delta,
                    constraint,
                    image_idx=image_idx,
                )
                self._wandb_run.log(
                    {f"samples/img_{image_idx}": wandb.Image(grid, caption=f"step={step}")}
                )

            t0 = time.perf_counter()
            x_protected, x_delta, _ = self.engine.protect(
                x_source=x_source,
                logging_hook=loss_logging_hook,
                validation_hook=validation_hook,
            )
            protect_elapsed = time.perf_counter() - t0

            save_tensor01_png(x_protected, self.out_dir / out_name)
            torch.save(x_delta.cpu(), self.out_dir / f"{f.stem}_delta.pt")
            n_written += 1
            protect_seconds.append(protect_elapsed)
            tqdm.write(
                f"[protection] {f.name} -> {out_name}: protect {format_duration(protect_elapsed)} "
                f"({protect_elapsed:.1f}s, {self.config.steps} steps)"
            )

        if n_in == 0:
            raise FileNotFoundError(f"No images in {self.dataset_images}")

        print(
            f"[protection] run_id={self.run_id} "
            f"dataset={derive_dataset_name(self.config.dataset_dir)} "
            f"images={n_in} newly_written={n_written} skipped_existing={n_skipped} "
            f"out_dir={self.out_dir}"
        )
        if protect_seconds:
            total_s = sum(protect_seconds)
            mean_s = total_s / len(protect_seconds)
            print(
                f"[protection] timing: n={len(protect_seconds)} "
                f"total={format_duration(total_s)} ({total_s:.1f}s) "
                f"mean={mean_s:.1f}s min={min(protect_seconds):.1f}s "
                f"max={max(protect_seconds):.1f}s"
            )
        if self._wandb_run is not None:
            self._wandb_run.log(
                {"run/images_total": n_in, "run/images_written": n_written}
            )
            self._wandb_run.finish()

    @torch.no_grad()
    def create_validation_grid(
        self,
        wrapper: DiTWrapper,
        x_protected,
        x_source,
        delta,
        constraint: PerturbationConstraint,
        *,
        image_idx: int,
    ) -> Image.Image:
        pil_image_protected = tensor01_to_pil(x_protected.to("cpu"))
        pil_image_clean = tensor01_to_pil(x_source.to("cpu"))

        mask_tensor = (
            constraint.epsilon_map.cpu()
            if constraint.epsilon_map is not None
            else torch.zeros(1, *x_protected.shape[-2:])
        )
        mask_np = (mask_tensor.squeeze(0) / constraint.eps).numpy()
        mask_colored = cm.inferno(mask_np)[..., :3]
        mask_pil = tensor01_to_pil(
            torch.from_numpy(mask_colored).permute(2, 0, 1).float().unsqueeze(0)
        )
        mask_labeled = add_label(mask_pil, "Epsilon Map")

        delta_tensor = constraint.lift(delta).cpu()
        delta_tensor = delta_tensor / (self.engine.eps * 2) + 0.5
        delta_labeled = add_label(tensor01_to_pil(delta_tensor.clamp(0, 1)), "Delta")

        if wrapper.model == ProtectionModel.FLUX2:
            editor = Flux2Editing(
                image_size=self.config.image_size,
                pipeline=wrapper.pipe,
            )
        elif wrapper.model == ProtectionModel.FIBO_EDIT:
            editor = FiboEditEditing(
                image_size=self.config.image_size,
                max_sequence_length=self.config.fibo_max_sequence_length,
                fibo_base_json=self.config.fibo_base_json,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported model for validation grid: {wrapper.model}")

        first_edit = self.config.surrogate_prompt
        edit_prompts = _VALIDATION_EDIT_PROMPTS.get(image_idx, [])
        editing_prompts = [first_edit, *edit_prompts]
        val_seed = self.config.seed + 1

        def run_edits(source_image: Image.Image, label_prefix: str) -> list[Image.Image]:
            results = []
            for prompt in editing_prompts:
                edited = editor.edit_image(
                    image=source_image,
                    prompt=prompt,
                    generator=torch.Generator(device=self.device).manual_seed(val_seed),
                )
                display = "uncond" if prompt == "" else prompt
                short = display if len(display) <= 42 else display[:42] + "…"
                results.append(add_label(edited, f"{label_prefix}\n{short}"))
            return results

        edited_protected = run_edits(pil_image_protected, "Protected")
        edited_clean = run_edits(pil_image_clean, "Original")
        protected_labeled = add_label(pil_image_protected, "Protected")
        clean_labeled = add_label(pil_image_clean, "Original")

        row1 = np.concatenate(
            [
                np.array(mask_labeled),
                np.array(protected_labeled),
                *[np.array(e) for e in edited_protected],
            ],
            axis=1,
        )
        row2 = np.concatenate(
            [
                np.array(delta_labeled),
                np.array(clean_labeled),
                *[np.array(e) for e in edited_clean],
            ],
            axis=1,
        )
        return Image.fromarray(np.concatenate([row1, row2], axis=0))


def run_protection(
    config: ProtectionConfig,
    *,
    source_config_path: Path | str | None = None,
) -> None:
    ProtectionRunner(config, source_config_path=source_config_path).run()
