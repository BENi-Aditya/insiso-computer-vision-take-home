#!/usr/bin/env python3
"""Convert MVTec AD metal_nut masks to YOLO boxes and split by source image."""
from __future__ import annotations
import argparse, csv, json, random, shutil
from pathlib import Path
import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def mask_boxes(mask_path: Path) -> list[tuple[int, int, int, int]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: raise ValueError(f"Could not read mask: {mask_path}")
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, area in stats[1:] if area]

def yolo_labels(boxes, width, height):
    rows=[]
    for x1,y1,x2,y2 in boxes:
        rows.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    return "\n".join(rows) + ("\n" if rows else "")

def image_records(raw: Path):
    category=raw/"metal_nut"
    if not category.exists(): raise FileNotFoundError(f"Expected MVTec category at {category}")
    records=[]
    for defect_dir in sorted((category/"test").iterdir()):
        if not defect_dir.is_dir(): continue
        for image in sorted(p for p in defect_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            boxes=[]
            if defect_dir.name != "good":
                mask=category/"ground_truth"/defect_dir.name/f"{image.stem}_mask.png"
                if not mask.exists(): raise FileNotFoundError(f"Missing ground-truth mask: {mask}")
                boxes=mask_boxes(mask)
            records.append({"image":image,"source":f"test/{defect_dir.name}/{image.name}","group":defect_dir.name,"boxes":boxes})
    for image in sorted(p for p in (category/"train"/"good").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        records.append({"image":image,"source":f"train/good/{image.name}","group":"good","boxes":[]})
    if not records: raise ValueError(f"No supported images found under {category}")
    return records

def split_records(records, seed, ratios=(.70,.15,.15)):
    rng=random.Random(seed); splits={name:[] for name in ("train","val","test")}; groups={}
    for row in records: groups.setdefault(row["group"],[]).append(row)
    for group, rows in sorted(groups.items()):
        rng.shuffle(rows); n=len(rows)
        if group!="good" and n<3: raise ValueError(f"Defect group '{group}' has only {n} examples; need >=3 for train/val/test")
        n_train=max(1,int(n*ratios[0])); n_val=max(1,int(n*ratios[1])) if n>=3 else 0
        n_train=min(n_train,n-n_val-(1 if group!="good" else 0)); n_test=n-n_train-n_val
        splits["train"].extend(rows[:n_train]); splits["val"].extend(rows[n_train:n_train+n_val]); splits["test"].extend(rows[n_train+n_val:n_train+n_val+n_test])
    return splits

def main():
    p=argparse.ArgumentParser(); p.add_argument("--raw",type=Path,default=Path("data/mvtec_anomaly_detection")); p.add_argument("--out",type=Path,default=Path("data/metal_nut_yolo")); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    if a.out.exists(): shutil.rmtree(a.out)
    a.out.mkdir(parents=True); records=image_records(a.raw); splits=split_records(records,a.seed)
    with (a.out/"manifest.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["split","source","group","boxes"]); writer.writeheader()
        for split, rows in splits.items():
            for row in rows:
                im=cv2.imread(str(row["image"]))
                if im is None: raise ValueError(f"Could not read image: {row['image']}")
                h,w=im.shape[:2]
                stem=Path(row["source"]).with_suffix("").as_posix().replace("/", "__")
                dest_img=a.out/"images"/split/f"{stem}{row['image'].suffix.lower()}"; dest_label=a.out/"labels"/split/f"{stem}.txt"
                dest_img.parent.mkdir(parents=True,exist_ok=True); dest_label.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(row["image"],dest_img); dest_label.write_text(yolo_labels(row["boxes"],w,h))
                writer.writerow({"split":split,"source":row["source"],"group":row["group"],"boxes":json.dumps(row["boxes"])})
    config={"path":str(a.out.resolve()),"train":"images/train","val":"images/val","test":"images/test","names":{0:"defect"}}
    (a.out/"data.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    audit=a.out/"label_audit"
    for group in sorted({r["group"] for r in records}):
        for row in [r for r in records if r["group"]==group][:3]:
            im=cv2.imread(str(row["image"]))
            for x1,y1,x2,y2 in row["boxes"]: cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
            audit.mkdir(parents=True,exist_ok=True)
            audit_stem=Path(row["source"]).with_suffix("").as_posix().replace("/", "__")
            cv2.imwrite(str(audit/f"{audit_stem}.jpg"),im)
    print(f"Wrote {len(records)} images to {a.out}; splits: "+", ".join(f"{k}={len(v)}" for k,v in splits.items()))
    print(f"Review overlays in {audit}; images were repartitioned by source image.")

if __name__=="__main__": main()
