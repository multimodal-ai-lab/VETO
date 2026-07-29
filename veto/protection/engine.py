import math
from typing import Callable

import torch
from torch import Tensor
from tqdm import tqdm

from veto.protection.constraints import (
    EpsilonConstraint,
    ImportanceEpsilonMap,
    ImportancePenalty,
    UnboundedConstraint,
    texture_map,
)
from veto.protection.objectives.base import ProtectionObjective


class PerturbationEngine:
    """Iterative input-space perturbation (PGD-style) to optimise a ``ProtectionObjective``."""

    def __init__(
        self,
        objective: ProtectionObjective,
        eps: float,
        alpha: float,
        steps: int,
        momentum_decay: float = 0.0,
        weight_decay: float = 0.0,
        use_sign: bool = True,
        seed: int = 0,
        constraint_type: str = "default_epsilon",
        constraint_threshold: float = None,
    ) -> None:
        self.objective = objective
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum_decay = momentum_decay
        self.weight_decay = weight_decay
        self.use_sign = use_sign
        self.constraint_threshold = constraint_threshold

        if constraint_type == "default_epsilon":
            self.constraint = EpsilonConstraint(eps=self.eps)
        elif constraint_type == "texture_penalty":
            self.constraint = ImportancePenalty(
                eps=self.eps,
                importance_fn=texture_map,
                weight=1000.0,
                threshold=constraint_threshold,
            )
        elif constraint_type == "texture_epsilon_map":
            self.constraint = ImportanceEpsilonMap(
                eps=self.eps,
                importance_fn=texture_map,
                threshold=constraint_threshold,
            )
        elif constraint_type == "unbounded":
            self.constraint = UnboundedConstraint()
        else:
            raise ValueError(f"Constraint type {constraint_type!r} not recognised.")

        print("Constraint:", self.constraint, self.constraint_threshold)

        self.alpha_schedule = lambda s: self.alpha
        self.seed = seed
        self.generator = torch.Generator(device="cuda:0").manual_seed(seed)

    def protect(
        self,
        x_source: Tensor,
        logging_hook: Callable = None,
        validation_hook: Callable = None,
    ) -> tuple[Tensor, Tensor, list[float]]:
        self.generator.manual_seed(self.seed)

        if hasattr(self.constraint, "prepare"):
            self.constraint.prepare(x_source[0])

        # Set the clean source in the objective
        if hasattr(self.objective, "prepare"):
            self.objective.prepare(x_source)

        self.objective.set_source(x_source)

        # PGD on delta
        delta = self.constraint.init_delta(x_source)
        momentum = torch.zeros_like(delta)
        losses: list[float] = []

        for step in tqdm(range(1, self.steps + 1)):

            # Optimizer zero_grad() equivalent
            delta.grad = None
            delta.requires_grad_(True)

            # Apply perturbation and clamp to valid RGB
            delta_lifted = self.constraint.lift(delta)
            x_protected = torch.clamp(x_source + delta_lifted, 0.0, 1.0)

            x_protected_aug = x_protected

            # ── Loss and gradient ──────────────────────────────────────────────
            # VETO returns a custom grad on x_protected_aug (internal backwards in objective).
            loss, x_protected_aug_grad = self.objective.loss(x_protected_aug)
            penalty = self.constraint.regularize(delta)

            if x_protected_aug_grad is None:
                # Standard backward through the full graph
                (loss + penalty).backward()
            else:
                # Objective already backpropped; inject its image grad through aug/clamp
                if penalty.requires_grad:
                    penalty.backward()
                x_protected_aug.backward(x_protected_aug_grad)

            loss_value = float(loss.detach().item())
            penalty_value = float(penalty.detach().item())
            current_alpha = self.alpha_schedule(step - 1)

            if logging_hook is not None:
                delta_full = self.constraint.lift(delta)
                saturation = (delta_full.abs() >= self.eps - 1e-7).float().mean().item()
                logging_hook(
                    loss_value,
                    penalty_value,
                    current_alpha,
                    saturation,
                    float(torch.abs(delta_full).max() * 255),
                    float(torch.abs(delta_full).mean() * 255),
                    float(torch.abs(momentum).max()),
                    float(torch.abs(momentum).mean()),
                    step=step,
                )

            # ── PGD step ─────────────────────────────────────────────────────
            with torch.no_grad():
                if delta.grad is None:
                    break

                grad = delta.grad.detach().contiguous()

                if self.use_sign:
                    if self.momentum_decay > 0.0:
                        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                        grad = grad / (grad_norm + 1e-8)
                        momentum = self.momentum_decay * momentum + grad
                        step_dir = momentum.sign()
                    else:
                        step_dir = grad.sign()
                else:
                    num_elements = delta[0].numel()
                    current_alpha = current_alpha * math.sqrt(num_elements)
                    if self.momentum_decay > 0.0:
                        grad_norm = torch.norm(
                            grad.view(grad.shape[0], -1), p=2, dim=1
                        ).view(-1, 1, 1, 1)
                        momentum = self.momentum_decay * momentum + grad / (grad_norm + 1e-8)
                        step_dir = momentum
                    else:
                        grad_norm = torch.norm(
                            grad.view(grad.shape[0], -1), p=2, dim=1
                        ).view(-1, 1, 1, 1)
                        step_dir = grad / (grad_norm + 1e-8)

                if self.weight_decay > 0.0:
                    delta = delta * self.weight_decay

                delta = delta - current_alpha * step_dir
                delta = self.constraint.project_delta(delta)

            delta = delta.detach()
            losses.append(loss_value)

            if validation_hook is not None:
                validation_hook(
                    wrapper=self.objective.wrapper,
                    x_protected=x_protected,
                    delta=delta,
                    constraint=self.constraint,
                    step=step,
                )

        delta_lifted = self.constraint.lift(delta)
        x_protected = torch.clamp(x_source + delta_lifted, 0.0, 1.0)
        return x_protected, delta, losses
