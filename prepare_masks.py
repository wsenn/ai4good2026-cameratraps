"""Segment Anything masks of the animals in the train crops, for `bg_swap: masks:`.

Each train crop of a split (`iwildcam.data.crop_task`) is segmented by SAM ViT-H (Kirillov et
al. 2023, github.com/facebookresearch/segment-anything) prompted with its MegaDetector box,
one mask per box (`multimask_output=False`; a box is not an ambiguous prompt). The mask is
written as `data/<crops>_sam/<crop name>.png` (0/255, crop size) only when SAM is confident
(predicted IoU >= --min-iou) and the mask covers a plausible part of the box (--min-fill);
`BackgroundSwapDataset` pastes the box rectangle for crops without one. `index.csv` lists every
crop's scores either way. SAM is not a dependency of the repo:

    uv run --with segment-anything==1.0 python prepare_masks.py \
        --checkpoint /zfs/ai4good/groups/camera-traps/groupB/models/sam/sam_vit_h_4b8939.pth

Reruns skip the crops already in `index.csv`.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from iwildcam.data import BOX_COLS, DATA, crop_task, image_dir_for, load_fold, load_task

SIDE = 1024                         # SAM's input side; the square crops need no padding


class Crops(Dataset):
    def __init__(self, names, boxes):
        self.names, self.boxes = names, boxes

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        im = Image.open(DATA / self.names[i]).convert("RGB")
        x = torch.from_numpy(np.asarray(im.resize((SIDE, SIDE), Image.BILINEAR)).copy())
        return x.permute(2, 0, 1).float(), torch.tensor(self.boxes[i] * SIDE), i


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, type=Path, help="SAM ViT-H weights")
    ap.add_argument("--split", default="official_ood", help="whose train crops to segment")
    ap.add_argument("--crops", default="crops_s448")
    ap.add_argument("--min-iou", type=float, default=0.88,     # below it: night / blur failures
                    help="SAM's predicted IoU to keep a mask")
    ap.add_argument("--min-fill", type=float, default=0.1, help="share of the box the mask covers")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=0, help="only the first N crops (a test)")
    a = ap.parse_args()
    from segment_anything import sam_model_registry

    task = crop_task(load_task(image_dir=image_dir_for(224), require_files=True), a.split, a.crops)
    rows = task.df.iloc[load_fold(task, a.split)["train"]]
    rows = rows[rows["side_h"].notna()].drop_duplicates("file_name")
    out = DATA / f"{a.crops}_sam"
    out.mkdir(exist_ok=True)
    index = out / "index.csv"
    if index.exists():
        rows = rows[~rows["file_name"].isin(pd.read_csv(index)["file_name"])]
    if a.limit:
        rows = rows.head(a.limit)
    print(f"{len(rows):,} {a.split} train crops to segment into {out}", flush=True)

    sam = sam_model_registry["vit_h"](checkpoint=str(a.checkpoint)).cuda().eval()
    pe = sam.prompt_encoder.get_dense_pe()
    loader = DataLoader(Crops(rows["file_name"].to_numpy(), rows[BOX_COLS[:4]].to_numpy()),
                        batch_size=a.batch_size, num_workers=a.workers)
    names = rows["file_name"].to_numpy()
    with ThreadPoolExecutor(8) as ex, tqdm(total=len(rows), unit="crop") as bar:
        for x, boxes, idx in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = sam.image_encoder(sam.preprocess(x.cuda()))
            recs = []
            for e, box, i in zip(emb, boxes.cuda(), idx.tolist()):
                sparse, dense = sam.prompt_encoder(points=None, boxes=box[None], masks=None)
                low, iou = sam.mask_decoder(image_embeddings=e[None].float(), image_pe=pe,
                                            sparse_prompt_embeddings=sparse,
                                            dense_prompt_embeddings=dense,
                                            multimask_output=False)
                with Image.open(DATA / names[i]) as im:
                    size = im.size
                m = (sam.postprocess_masks(low, (SIDE, SIDE), size[::-1])[0, 0] > 0).cpu().numpy()
                x0, y0, x1, y1 = (box.cpu().numpy() / SIDE * size[0]).round().astype(int)
                inside = m[y0:y1, x0:x1].sum()
                fill = inside / max(1, (x1 - x0) * (y1 - y0))
                spill = 1 - inside / max(1, m.sum())
                keep = bool(iou.item() >= a.min_iou and fill >= a.min_fill)
                if keep:
                    dst = out / (Path(names[i]).stem + ".png")
                    ex.submit(Image.fromarray(m.astype(np.uint8) * 255).save, dst)
                recs.append({"file_name": names[i], "iou": round(iou.item(), 4),
                             "fill": round(float(fill), 4), "spill": round(float(spill), 4),
                             "kept": keep})
            pd.DataFrame(recs).to_csv(index, mode="a", header=not index.exists(), index=False)
            bar.update(len(recs))
    s = pd.read_csv(index)
    print(f"{s.kept.sum():,} of {len(s):,} crops have a mask ({s.kept.mean():.0%}); "
          f"median IoU {s.iou.median():.2f}, fill {s.fill.median():.2f}")


if __name__ == "__main__":
    main()
