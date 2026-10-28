"""Save FDA examples for a visual check: photo | donor (another camera) | result.

    python -m scripts.fda_examples --beta 0.01 --n 6 --out ~/analysis/2026-10-29-fda-examples
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from iwildcam.data import image_dir_for, load_fold, load_task
from iwildcam.experiments import fda_swap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    task = load_task(image_dir=image_dir_for(224))
    train = load_fold(task, "official_ood")["train"]
    rng = random.Random(a.seed)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sz = (a.size, a.size)
    for n in range(a.n):
        i = rng.choice(train)
        j = rng.choice(train)
        while task.camera[j] == task.camera[i]:
            j = rng.choice(train)
        src, ref = (Image.open(task.image_dir / task.df["file_name"].iloc[k]).convert("RGB")
                    .resize(sz, Image.BILINEAR) for k in (i, j))
        res = Image.fromarray(fda_swap(np.asarray(src), np.asarray(ref), a.beta))
        strip = Image.new("RGB", (3 * a.size + 8, a.size), "white")
        for col, im in enumerate((src, ref, res)):
            strip.paste(im, (col * (a.size + 4), 0))
        name = (f"{n}_cam{task.camera[i]}-{task.df['category'].iloc[i]}"
                f"_donor-cam{task.camera[j]}_beta{a.beta}.jpg")
        strip.save(out / name, quality=90)
        print(out / name)


if __name__ == "__main__":
    main()
