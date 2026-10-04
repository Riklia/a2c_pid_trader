"""Prints every input the pipeline uses, where it comes from and how it is normalized.

Run from repo root:
    python scripts/describe_features.py configs/walk_forward.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import model_features
from models.pid_net import PIDNetInputBuilder


def main(cfg_path: Path) -> None:
    with open(cfg_path) as f:
        cfg = json.load(f)
    specs = model_features.parse_feature_specs(cfg["a2c_pid"]["features"])
    missing = cfg["a2c_pid"]["cross_asset_missing"]
    if missing == "fill":
        specs = model_features.with_availability_flags(specs)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 200)

    print(f"traded symbol: {cfg['data']['symbol']}\n")

    print("PID law (static_pid and the static part of a2c_pid), raw values, not configurable:")
    for column in ["e_p", "e_i", "e_d"]:
        print(f"  {column:4s} {model_features.COLUMN_DESCRIPTIONS[column]}")

    n_inputs = PIDNetInputBuilder.input_size(len(specs))
    n_flags = sum(spec.column == model_features.AVAILABLE for spec in specs)
    print(f"\nPIDNet input ({n_inputs} values = 2 past positions + {len(specs) - n_flags} features from "
          f"a2c_pid.features + {n_flags} automatic availability flags):")
    print("  u(k-1), u(k-2): the controller's own last two positions, raw")
    print(model_features.describe_features(specs, cfg['data']['symbol']).to_string(index=False))
    print("\n  normalizations:")
    for norm in dict.fromkeys(spec.norm for spec in specs):
        print(f"    {norm}: {model_features.NORMALIZATIONS[norm]}")

    others = model_features.other_symbols(specs)
    if others:
        print(f"\ncross-asset symbols: {', '.join(others)}")
        print(f"bars where one of them has no value ({missing!r}): {model_features.MISSING_POLICIES[missing]}")

    print("\nactor / critic: [position, realized_vol, realized_vol], raw (training only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    main(parser.parse_args().config)
