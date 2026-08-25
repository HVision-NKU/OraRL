"""Console entry point for the portable data builder."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Load the data builder lazily so CLI help stays lightweight."""

    from orarl.data.build import main as build_main

    result = build_main(argv)
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
