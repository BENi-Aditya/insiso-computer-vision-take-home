#!/usr/bin/env python3
"""Train a small YOLO detector. Use a separate --name for each experiment."""
import argparse, os
from pathlib import Path
config_dir = Path(__file__).resolve().parents[1] / ".ultralytics"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
matplotlib_dir = Path(__file__).resolve().parents[1] / ".matplotlib"
matplotlib_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/metal_nut_yolo/data.yaml"); p.add_argument("--weights",default="yolo11n.pt"); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch",type=int,default=8); p.add_argument("--seed",type=int,default=42); p.add_argument("--name",default="baseline_640"); p.add_argument("--device",default="mps"); a=p.parse_args()
    if not Path(a.data).exists(): p.error(f"Dataset config not found: {a.data}; run scripts/prepare_data.py first")
    YOLO(a.weights).train(data=a.data,imgsz=a.imgsz,epochs=a.epochs,batch=a.batch,seed=a.seed,deterministic=True,device=a.device,workers=0,project="runs/detect",name=a.name,patience=20,plots=True,pretrained=True,exist_ok=True)
    print(f"Best checkpoint: runs/detect/{a.name}/weights/best.pt")
if __name__=="__main__": main()
