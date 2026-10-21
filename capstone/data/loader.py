"""Load the synthetic world from disk (generating it on first use)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(__file__).parent / "synthetic" / "out"


def ensure_world(path: Path = DEFAULT_DIR, small: bool = False) -> Path:
    if not (path / "tenants.jsonl").exists():
        from capstone.data.synthetic.generator import Generator, write_world

        gen = Generator(customers_per_tenant=8, jobs_per_tenant=60, transcripts_per_tenant=30) \
            if small else Generator()
        write_world(gen.build(), path)
    return path


@lru_cache(maxsize=32)
def load_table(name: str, path: Path = DEFAULT_DIR) -> list[dict[str, Any]]:
    ensure_world(path)
    with (path / f"{name}.jsonl").open() as fh:
        return [json.loads(line) for line in fh]


def load_frame(name: str, path: Path = DEFAULT_DIR):  # -> pandas.DataFrame
    import pandas as pd

    return pd.DataFrame(load_table(name, path))
