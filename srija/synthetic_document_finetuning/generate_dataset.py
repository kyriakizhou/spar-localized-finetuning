from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdf_dataset import SEED, materialize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic document finetuning dataset.")
    parser.add_argument("--output-dir", default="datasets", help="Directory for generated dataset files")
    parser.add_argument("--docs-per-belief", type=int, default=80, help="Number of SDF documents per belief")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of documents held out")
    parser.add_argument("--seed", type=int, default=SEED, help="Deterministic shuffle/rendering seed")
    args = parser.parse_args()

    summary = materialize_dataset(
        output_dir=Path(args.output_dir),
        docs_per_belief=args.docs_per_belief,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

