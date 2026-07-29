import argparse

from veto.protection.config_loader import load_protection_config
from veto.protection.runner import run_protection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Veto image protection against unauthorized image editing."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/protection/flux2.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    cfg = load_protection_config(args.config)
    run_protection(cfg, source_config_path=args.config)


if __name__ == "__main__":
    main()
