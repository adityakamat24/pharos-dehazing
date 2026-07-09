"""CLI for phase-2 pseudo-labeling via the restoration teacher ensemble.

Runs the ensemble over a directory of real (unpaired) frames, writing the best
candidate per image plus a JSON manifest. See pharos.teachers.ensemble.

Example:
    python scripts/pseudo_label.py --in data/rtts/frames --out data/pseudo/rtts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a bare script (python scripts/pseudo_label.py ...).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pharos.teachers.ensemble import NoRefScorer, RestorationEnsemble  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Pharos restoration-ensemble pseudo-labeler.")
    ap.add_argument("--in", dest="dir_in", required=True, help="input image directory")
    ap.add_argument("--out", dest="dir_out", required=True, help="output directory (best + manifest.json)")
    ap.add_argument("--brisque-model", default=None, help="optional cv2.quality BRISQUE model yml")
    ap.add_argument("--brisque-range", default=None, help="optional cv2.quality BRISQUE range yml")
    args = ap.parse_args(argv)

    scorer = NoRefScorer(brisque_model=args.brisque_model, brisque_range=args.brisque_range)
    ensemble = RestorationEnsemble(scorer=scorer)
    manifest = ensemble.run(args.dir_in, args.dir_out)
    print(
        f"pseudo-labeled {manifest['num_images']} images "
        f"(scorer={manifest['scorer']}, members={manifest['members']}) -> {args.dir_out}"
    )


if __name__ == "__main__":
    main()
