from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdf_selective_dataset import SEED, materialize_all


def parse_domains(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the SDF selective-facts pilot dataset.")
    parser.add_argument("--output-root", default="dataset", help="Directory for materialized dataset artifacts.")
    parser.add_argument("--domains", default=None, help="Optional comma-separated domain subset for debug builds.")
    parser.add_argument("--facts-per-domain", type=int, default=None, help="Optional even number of facts per domain.")
    parser.add_argument("--max-docs-per-fact", type=int, default=None, help="Optional cap for debug builds.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    manifest = materialize_all(
        output_root=Path(args.output_root),
        domains=parse_domains(args.domains),
        facts_per_domain=args.facts_per_domain,
        max_docs_per_fact=args.max_docs_per_fact,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
