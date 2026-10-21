"""Mirror local prompt versions + tags to the LangSmith prompt registry.

    uv run python scripts/push_prompts.py                    # push all versions, apply tags
    uv run python scripts/push_prompts.py --promote staging  # only move the staging tag
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langsmith import Client

from capstone.prompts.registry import PROMPT_DIR, _load, build_template


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", help="only re-apply this tag")
    args = ap.parse_args()
    client = Client()
    for path in sorted(Path(PROMPT_DIR).glob("*.yaml")):
        spec = _load(path.stem)
        tags_by_version: dict[str, list[str]] = {}
        for tag, version in spec.get("tags", {}).items():
            if args.promote and tag != args.promote:
                continue
            tags_by_version.setdefault(version, []).append(tag)
        for version, vspec in spec["versions"].items():
            if args.promote and version not in tags_by_version:
                continue
            url = client.push_prompt(
                spec["name"],
                object=build_template(vspec, with_history=True),
                description=spec.get("description"),
                commit_tags=[version, *tags_by_version.get(version, [])],
            )
            print(f"pushed {spec['name']} {version} tags={tags_by_version.get(version, [])} {url}")


if __name__ == "__main__":
    main()
