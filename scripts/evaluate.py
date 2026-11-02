#!/usr/bin/env python3
"""Held-out single-class detection metrics and real error images."""
import argparse, csv, json, os
from pathlib import Path
import cv2
import numpy as np
config_dir = Path(__file__).resolve().parents[1] / ".ultralytics"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
matplotlib_dir = Path(__file__).resolve().parents[1] / ".matplotlib"
matplotlib_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
from ultralytics import YOLO

def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/(aa+bb-inter) if aa+bb>inter else 0.

def boxes_from_label(path,w,h):
    rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            _,cx,cy,bw,bh=map(float,line.split()); rows.append(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h))
    return rows

def ap101(tp,fp,positives):
    if positives==0: return 0.
    tp,fp=np.cumsum(tp),np.cumsum(fp); recall=tp/positives; precision=tp/np.maximum(tp+fp,1e-12)
    return float(np.mean([precision[recall>=r].max(initial=0.) for r in np.linspace(0,1,101)]))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--weights",required=True); p.add_argument("--data",default="data/metal_nut_yolo"); p.add_argument("--split",default="test",choices=("val","test")); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--conf",type=float,default=.25); p.add_argument("--iou-match",type=float,default=.5); p.add_argument("--out",default="results/generated/baseline"); p.add_argument("--device",default="mps"); a=p.parse_args()
    root,out=Path(a.data),Path(a.out); out.mkdir(parents=True,exist_ok=True); images=sorted((root/"images"/a.split).glob("*"))
    if not images: p.error(f"No images in {root}/images/{a.split}")
    model=YOLO(a.weights); entries=[]; total_gt=0; per_image=[]; failures=[]
    for path in images:
        image=cv2.imread(str(path))
        if image is None: raise ValueError(f"Cannot read {path}")
        h,w=image.shape[:2]; gt=boxes_from_label(root/"labels"/a.split/f"{path.stem}.txt",w,h); total_gt+=len(gt)
        pred=model.predict(str(path),imgsz=a.imgsz,conf=.001,device=a.device,verbose=False)[0]
        boxes=[(list(map(float,b.xyxy[0].cpu().numpy())),float(b.conf[0])) for b in pred.boxes]
        entries.extend((score,path,box) for box,score in boxes)
        chosen=sorted([(score,box) for box,score in boxes if score>=a.conf],key=lambda z:z[0],reverse=True); used=set(); tp=fp=0
        for score,box in chosen:
            candidates=[(iou(box,g),i) for i,g in enumerate(gt) if i not in used]; overlap,best=max(candidates,default=(0.,-1))
            if overlap>=a.iou_match: used.add(best); tp+=1
            else: fp+=1
        misses=[g for i,g in enumerate(gt) if i not in used]
        row={"image":path.name,"ground_truth":len(gt),"predictions_at_threshold":len(chosen),"true_positives":tp,"false_positives":fp,"misses":len(misses)}; per_image.append(row)
        if fp or misses: failures.append((path,gt,chosen,misses,row))
    # Confidence-ranked, greedy one-to-one matching for the single defect class.
    entries.sort(key=lambda z:z[0],reverse=True); used={path:set() for path in images}; tp_rank=[]; fp_rank=[]
    gt_map={}
    for path in images:
        im=cv2.imread(str(path)); gt_map[path]=boxes_from_label(root/"labels"/a.split/f"{path.stem}.txt",im.shape[1],im.shape[0])
    for _,path,box in entries:
        candidates=[(iou(box,g),i) for i,g in enumerate(gt_map[path]) if i not in used[path]]; overlap,best=max(candidates,default=(0.,-1))
        if overlap>=a.iou_match: tp_rank.append(1); fp_rank.append(0); used[path].add(best)
        else: tp_rank.append(0); fp_rank.append(1)
    n_tp=sum(r["true_positives"] for r in per_image); n_fp=sum(r["false_positives"] for r in per_image)
    metrics={"split":a.split,"images":len(images),"ground_truth_boxes":total_gt,"confidence_threshold":a.conf,"iou_match_threshold":a.iou_match,"precision":n_tp/(n_tp+n_fp) if n_tp+n_fp else 0.,"recall":n_tp/total_gt if total_gt else 0.,"mAP50":ap101(tp_rank,fp_rank,total_gt),"ap_method":"single-class 101-point interpolated AP; detections ranked by confidence; IoU=0.5"}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    with (out/"per_image.csv").open("w",newline="") as f: writer=csv.DictWriter(f,fieldnames=per_image[0].keys()); writer.writeheader(); writer.writerows(per_image)
    faildir=out/"failure_cases"; faildir.mkdir(exist_ok=True)
    with (faildir/"index.csv").open("w",newline="") as f:
        fields=["image","ground_truth","predictions_at_threshold","true_positives","false_positives","misses","ground_truth_boxes_xyxy","predictions_xyxy_confidence","missed_ground_truth_boxes_xyxy","manual_analysis"]; writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for path,gt,chosen,misses,row in failures:
            im=cv2.imread(str(path))
            for b in gt: cv2.rectangle(im,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,255,0),2)
            for score,b in chosen: cv2.rectangle(im,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,0,255),2)
            cv2.imwrite(str(faildir/path.name),im)
            writer.writerow({**row,"ground_truth_boxes_xyxy":json.dumps(gt),"predictions_xyxy_confidence":json.dumps([{"xyxy":b,"confidence":score} for score,b in chosen]),"missed_ground_truth_boxes_xyxy":json.dumps(misses),"manual_analysis":"REVIEW: green=ground truth, red=prediction; record visual cause"})
    print(json.dumps(metrics,indent=2)); print(f"Genuine error images: {len(failures)}; inspect {faildir}/index.csv")
if __name__=="__main__": main()
