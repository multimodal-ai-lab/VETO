from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Texture utility  (one concrete source of an importance map)
# ---------------------------------------------------------------------------

def texture_map(
        image: Tensor,
        hf_sigma: float = 1.5,
        pool_window: int = 15,  # was 31 before
        border: int = 16,
) -> Tensor:
    """
    Returns a texture intensity map of shape (1, H, W), normalised to [0, 1].

    Smooth regions → 0, textured regions → 1.
    Border pixels are forced to 0 to suppress edge artefacts.

    This is one way to derive an importance map for use with
    :class:`ImportancePenalty` or :class:`ImportanceEpsilonMap`.

    Args:
        image: RGB tensor of shape (3, H, W).
    """
    device = image.device

    lum = (
            0.2126 * image[0:1] +
            0.7152 * image[1:2] +
            0.0722 * image[2:3]
    ).unsqueeze(0)  # (1, 1, H, W)

    def _gaussian_kernel(sigma: float) -> Tensor:
        size = int(6 * sigma + 1) | 1
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g /= g.sum()
        return (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)

    kernel = _gaussian_kernel(hf_sigma)
    hf = lum - F.conv2d(lum, kernel, padding=kernel.shape[-1] // 2)

    box = torch.ones(1, 1, pool_window, pool_window, device=device) / pool_window ** 2
    local_rms = F.conv2d(hf ** 2, box, padding=pool_window // 2).sqrt()

    lo, hi = local_rms.amin(), local_rms.amax()
    out = ((local_rms - lo) / (hi - lo + 1e-8)).squeeze(0)  # (1, H, W)

    out[..., :border, :] = 0.0
    out[..., -border:, :] = 0.0
    out[..., :, :border] = 0.0
    out[..., :, -border:] = 0.0

    return out


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class PerturbationConstraint(ABC):
    """
    Defines the geometry of the perturbation search space.

    Subclasses implement ``project_delta`` to enforce the feasible set.
    ``init_delta`` and ``regularize`` have sensible defaults and only need
    overriding in special cases.

    Constraints that depend on the source image implement ``prepare(x)``,
    which the protection loop calls once before optimisation begins.
    """

    def __init__(self, eps: float):
        self.eps = eps

    def init_delta(self, x: Tensor) -> Tensor:
        """Initialise δ to zero. Override for e.g. random PGD restarts."""
        return torch.zeros_like(x)

    @abstractmethod
    def project_delta(self, delta: Tensor) -> Tensor: ...

    def prepare(self, x: Tensor) -> None:
        """Pre-compute image-dependent state. No-op by default."""

    def regularize(self, delta: Tensor) -> Tensor:
        """Scalar penalty added to the protection loss. Zero by default."""
        return delta.sum() * 0.0

    def lift(self, delta: Tensor) -> Tensor:
        """Map from the constraint's native space to pixel space. Identity by default."""
        return delta

    @property
    def epsilon_map(self) -> Tensor | None:
        """Per-pixel epsilon budget, or ``None`` if the budget is uniform."""
        return None


class ImportanceConstraint(PerturbationConstraint, ABC):
    """
    Base for constraints driven by a per-pixel importance map.

    ``importance_fn`` maps a source image to a (1, H, W) tensor in [0, 1],
    where higher values indicate regions that should receive more perturbation
    budget (e.g. textured, low-saliency, or background regions).

    Any callable with that signature works: texture energy, inverted saliency,
    depth-based masks, segmentation confidence, and so on.

    During ``prepare()``, this class builds a per-pixel epsilon map from the
    importance scores using one of two redistribution modes controlled by
    ``threshold``:

    * ``threshold=None`` *(smooth)*: ``epsilon_map`` is proportional to the
      continuous importance score, renormalised so its mean equals ``eps`` and
      clamped to ``[eps * eps_min_ratio, eps * eps_max_ratio]``.
    * ``threshold=float`` *(hard)*: each pixel receives one of exactly two
      budgets — ``eps`` where importance meets the threshold, or
      ``eps * low_importance_ratio`` elsewhere.

    Subclasses consume ``epsilon_map`` differently: :class:`ImportanceEpsilonMap`
    enforces it geometrically via projection; :class:`ImportancePenalty` inverts
    it into a loss penalty while keeping the projection uniform.

    Args:
        eps:                  Mean L∞ budget across the image.
        importance_fn:        Maps (3, H, W) → (1, H, W) importance map in [0, 1].
        threshold:            Binarisation threshold. ``None`` for continuous map.
        eps_min_ratio:        Floor for per-pixel eps as a fraction of ``eps``.
                              Only used when ``threshold=None``.
        eps_max_ratio:        Ceiling for per-pixel eps as a fraction of ``eps``.
                              Only used when ``threshold=None``.
        low_importance_ratio: Budget fraction for below-threshold pixels.
                              Only used when ``threshold`` is set.
    """

    def __init__(
            self,
            eps: float,
            importance_fn: Callable[[Tensor], Tensor],
            threshold: float | None = None,
            eps_min_ratio: float = 0.0,
            eps_max_ratio: float = 1.0,
            low_importance_ratio: float = 0.0,
    ):
        super().__init__(eps)
        self.importance_fn = importance_fn
        self.threshold = threshold
        self.eps_min_ratio = eps_min_ratio
        self.eps_max_ratio = eps_max_ratio
        self.low_importance_ratio = low_importance_ratio
        self._epsilon_map: Tensor | None = None

    def prepare(self, x: Tensor) -> None:
        importance = self.importance_fn(x)
        if self.threshold is None:
            raw = self.eps * importance / (importance.mean(dim=(1, 2), keepdim=True) + 1e-8)
            self._epsilon_map = raw.clamp(
                self.eps * self.eps_min_ratio,
                self.eps * self.eps_max_ratio,
            )
        else:
            self._epsilon_map = torch.where(
                importance >= self.threshold,
                torch.full_like(importance, self.eps),
                torch.full_like(importance, self.eps * self.low_importance_ratio),
            )

    def _get_epsilon_map(self) -> Tensor:
        if self._epsilon_map is None:
            raise RuntimeError("Call prepare(x) before the optimisation loop.")
        return self._epsilon_map

    @property
    def epsilon_map(self) -> Tensor | None:
        return self._epsilon_map


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class UnboundedConstraint(PerturbationConstraint):
    """No projection — δ is free. Useful for debugging or ablation studies."""

    def __init__(self):
        super().__init__(eps=1.0)  # 1.0 corresponds to 255 / 255

    def project_delta(self, delta: Tensor) -> Tensor:
        return delta


class EpsilonConstraint(PerturbationConstraint):
    """
    Uniform L∞ ball of radius ``eps``.

    Every pixel is clipped to [-eps, eps] independently.
    This is the standard fixed-budget baseline.
    """

    def project_delta(self, delta: Tensor) -> Tensor:
        return delta.clamp(-self.eps, self.eps)


# ---------------------------------------------------------------------------
# Importance-map constraints
# ---------------------------------------------------------------------------

class ImportancePenalty(ImportanceConstraint):
    """
    Uniform L∞ projection + a loss penalty that discourages perturbations in
    low-importance regions.

    The projection allows every pixel to reach ±eps; the penalty steers the
    optimiser toward high-importance regions through the loss rather than by
    narrowing the feasible set.  This makes the two classes conceptually dual:
    :class:`ImportanceEpsilonMap` enforces the budget geometrically,
    ``ImportancePenalty`` enforces it through the loss.

    The penalty mask is derived by inverting ``epsilon_map``:

    * ``threshold=None`` *(soft)*: penalty weight at pixel p is
      ``1 - epsilon_map[p] / (eps * eps_max_ratio)``, tapering smoothly from
      1 (lowest-budget pixels) toward 0 (highest-budget pixels).
    * ``threshold=float`` *(hard)*: pixels that received the reduced budget
      (``epsilon_map < eps``) are fully penalised; others receive zero penalty.

    Args:
        eps:    Global L∞ bound applied during projection.
        weight: Scalar multiplier for the penalty term.
        importance_fn, threshold, eps_min_ratio, eps_max_ratio,
        low_importance_ratio: see :class:`ImportanceConstraint`.
    """

    def __init__(
            self,
            eps: float,
            weight: float,
            importance_fn: Callable[[Tensor], Tensor],
            threshold: float | None = None,
            eps_min_ratio: float = 0.0,
            eps_max_ratio: float = 2.0,
            low_importance_ratio: float = 0.0,
    ):
        super().__init__(eps, importance_fn, threshold, eps_min_ratio, eps_max_ratio, low_importance_ratio)
        self.weight = weight

    def _penalty_mask(self) -> Tensor:
        epsilon_map = self._get_epsilon_map()
        if self.threshold is None:
            return 1.0 - epsilon_map / (self.eps * self.eps_max_ratio)
        else:
            return (epsilon_map < self.eps).float()

    def project_delta(self, delta: Tensor) -> Tensor:
        return delta.clamp(-self.eps, self.eps)

    def regularize(self, delta: Tensor) -> Tensor:
        return self.weight * (delta * self._penalty_mask()).pow(2).mean()


class ImportanceEpsilonMap(ImportanceConstraint):
    """
    Per-pixel L∞ projection with a budget proportional to the importance map.

    The projection enforces a per-pixel bound geometrically — no loss term is
    involved.  High-importance pixels receive more headroom; low-importance
    pixels receive less.

    See :class:`ImportanceConstraint` for the two redistribution modes
    (``threshold=None`` for smooth, ``threshold=float`` for hard step).
    """

    def project_delta(self, delta: Tensor) -> Tensor:
        epsilon_map = self._get_epsilon_map()
        return delta.clamp(-epsilon_map, epsilon_map)