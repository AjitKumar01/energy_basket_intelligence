#!/usr/bin/env python3
"""Run the generic held-out price-choice experiment from a dataset configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/version4"))

from external_choice_adapters import adapter_from_config
from external_choice_pipeline import ExternalChoiceExperiment, ValidationSettings


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/external_price_validation.json",
        help="JSON file listing dataset adapters, sources, and common audit settings",
    )
    parser.add_argument("--output", type=Path, help="override configured JSON output path")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    configuration = json.loads(config_path.read_text())
    if configuration.get("schema_version") != 1:
        raise ValueError("external validation config requires schema_version=1")
    specifications = configuration.get("datasets")
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("external validation config requires a nonempty datasets list")

    settings_values = dict(configuration.get("validation", {}))
    if "regularization_grid" in settings_values:
        settings_values["regularization_grid"] = tuple(
            float(value) for value in settings_values["regularization_grid"])
    settings = ValidationSettings(**settings_values)
    adapters = [
        adapter_from_config(specification, base_directory=config_path.parent)
        for specification in specifications
    ]
    result = ExternalChoiceExperiment(settings).run(adapters)
    result["configuration"] = str(config_path)

    configured_output = configuration.get(
        "output", "../artifacts/external_price_validation_audited.json")
    output = args.output.resolve() if args.output else _resolve(
        config_path.parent, configured_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
