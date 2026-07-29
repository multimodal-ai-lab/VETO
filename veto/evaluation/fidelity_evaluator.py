import math
from typing import Dict, NamedTuple, Tuple

import lpips
import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim

from veto.utils.images import resize_image


class FidelityScores(NamedTuple):
    mse: float
    psnr: float
    ssim: float
    lpips: float


class FidelityEvaluator:
    """
    Compare an original image and a protected image.
    """

    def __init__(
        self,
        device: torch.device,
        image_size: int = 512,
    ) -> None:
        self.device = device
        self.image_size: Tuple[int, int] = (image_size, image_size)
        self._lpips = lpips.LPIPS(net="alex").to(device).eval()

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = resize_image(image, self.image_size[0])
        t = torch.from_numpy(np.array(image)).float() / 255.0
        if t.dim() == 3:
            t = t.permute(2, 0, 1)
        return t.unsqueeze(0).to(self.device)

    def compute_mse(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        return torch.nn.functional.mse_loss(img1, img2).item()

    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        mse = self.compute_mse(img1, img2)
        if mse <= 0:
            return float("inf")
        return 10.0 * math.log10(1.0 / mse)

    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        return float(ssim(img1, img2, data_range=1.0).item())

    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        o = img1 * 2.0 - 1.0
        p = img2 * 2.0 - 1.0
        with torch.no_grad():
            return float(self._lpips(o, p).item())

    def __call__(self, original: Image.Image, protected: Image.Image) -> Dict[str, float]:
        img1 = self._preprocess_image(original)
        img2 = self._preprocess_image(protected)

        return {
            "mse": self.compute_mse(img1, img2),
            "psnr": self.compute_psnr(img1, img2),
            "ssim": self.compute_ssim(img1, img2),
            "lpips": self.compute_lpips(img1, img2),
        }

    def evaluate(self, original_path: str, protected_path: str) -> FidelityScores:
        original_pil = Image.open(original_path).convert("RGB")
        protected_pil = Image.open(protected_path).convert("RGB")
        metrics = self.__call__(original_pil, protected_pil)
        return FidelityScores(
            mse=metrics["mse"],
            psnr=metrics["psnr"],
            ssim=metrics["ssim"],
            lpips=metrics["lpips"],
        )

    def unload(self) -> None:
        del self._lpips
