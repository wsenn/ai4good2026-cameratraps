"""Write a `train_weights` CSV: the labelled TRAIN photos of `official_ood` or
`random_burst` with no MegaDetector v3 box at or above `--conf`, each with `--weight`.

    python -m scripts.make_nobox_weights --weight 0.2 --out nobox_w0.2.csv

Weight 0 removes these photos from training; photos not in the file keep weight 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from iwildcam.data import EMPTY, load_metadata

MD_JSON = Path("/zfs/ai4good/groups/camera-traps/groupB/datasets/iWildCam2020/"
               "iwildcam2020_megadetector_results.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weight", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.6, help="minimum box confidence")
    ap.add_argument("--md-json", default=str(MD_JSON))
    a = ap.parse_args()

    images = json.loads(Path(a.md_json).read_text())["images"]
    with_box = {im["id"] for im in images
                if any(d["conf"] >= a.conf for d in im.get("detections") or [])}
    meta = load_metadata()
    train = (meta["official_ood"] == "train") | (meta["random_burst"] == "train")
    rows = meta[train & (meta["category"] != EMPTY)]
    nobox = rows[~rows["image_id"].isin(with_box)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": nobox["image_id"], "weight": a.weight}).to_csv(a.out, index=False)
    known = {im["id"] for im in images}
    print(f"{len(nobox):,} of {len(rows):,} labelled train photos have no box >= {a.conf} "
          f"({(~nobox['image_id'].isin(known)).sum()} not in {Path(a.md_json).name}); "
          f"weight {a.weight} -> {a.out}")


if __name__ == "__main__":
    main()
