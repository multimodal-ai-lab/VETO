from pathlib import Path

_DATASET_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
PROTECTED_OUTPUT_EXT = ".png"


def protected_output_name(source_filename: str) -> str:
    return f"{Path(source_filename).stem}{PROTECTED_OUTPUT_EXT}"


def resolve_dataset_images_dir(dataset_dir: Path) -> Path:
    dataset_dir = Path(dataset_dir)
    images_root = dataset_dir / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(
            f"Dataset images layout not found (expected {images_root} or "
            f"{images_root / 'base'}): {images_root}"
        )
    has_direct_images = any(
        p.is_file() and p.suffix.lower() in _DATASET_IMAGE_EXTS
        for p in images_root.iterdir()
    )
    if has_direct_images:
        return images_root
    base_dir = images_root / "base"
    if base_dir.is_dir():
        return base_dir
    raise FileNotFoundError(
        f"No image files under {images_root} and no {base_dir} directory."
    )


def derive_dataset_name(dataset_path: Path) -> str:
    return Path(dataset_path).name


def images_dataset_root(output_root: Path, dataset_name: str) -> Path:
    return Path(output_root) / "images" / dataset_name


def images_root(output_root: Path, dataset_name: str, run_id: str) -> Path:
    return images_dataset_root(output_root, dataset_name) / run_id


def protection_config_path(output_root: Path, dataset_name: str, run_id: str) -> Path:
    return images_root(output_root, dataset_name, run_id) / "config.yaml"


def protected_images_dir(output_root: Path, dataset_name: str, run_id: str) -> Path:
    return images_root(output_root, dataset_name, run_id) / "protected" / "images"



def edited_unprotected_dir(output_root: Path, dataset_name: str, edit_model: str) -> Path:
    return images_dataset_root(output_root, dataset_name) / "edited_unprotected" / edit_model


def evaluation_images_root(
    output_root: Path,
    dataset_name: str,
    run_id: str,
    variant: str = "base",
) -> Path:
    return images_root(output_root, dataset_name, run_id) / "evaluations" / variant


def edited_variant_dir(
    output_root: Path,
    dataset_name: str,
    run_id: str,
    edit_model: str,
    variant: str = "base",
) -> Path:
    return (
        evaluation_images_root(output_root, dataset_name, run_id, variant)
        / "edited"
        / edit_model
        / "images"
    )


def results_dir(output_root: Path, dataset_name: str, run_id: str) -> Path:
    return Path(output_root) / "results" / dataset_name / run_id


def evaluation_results_dir(
    output_root: Path,
    dataset_name: str,
    run_id: str,
    variant: str = "base",
) -> Path:
    return results_dir(output_root, dataset_name, run_id) / "evaluations" / variant


def metrics_dir(output_root: Path, dataset_name: str, run_id: str) -> Path:
    return Path(output_root) / "metrics" / dataset_name / run_id


def evaluation_metrics_dir(
    output_root: Path,
    dataset_name: str,
    run_id: str,
    variant: str = "base",
) -> Path:
    return metrics_dir(output_root, dataset_name, run_id) / "evaluations" / variant
