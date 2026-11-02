#!/usr/bin/env python3
import argparse, os
from pathlib import Path
config_dir = Path(__file__).resolve().parents[1] / ".ultralytics"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
matplotlib_dir = Path(__file__).resolve().parents[1] / ".matplotlib"
matplotlib_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument("--weights",required=True); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--out",default="models/metal_nut_defect.onnx"); a=p.parse_args()
target=Path(a.out); target.parent.mkdir(parents=True,exist_ok=True); exported=Path(YOLO(a.weights).export(format="onnx",imgsz=a.imgsz,opset=12,simplify=False,device="cpu"))
if exported.resolve()!=target.resolve(): exported.replace(target)
print(f"ONNX export written to {target} ({target.stat().st_size/1e6:.1f} MB)")
