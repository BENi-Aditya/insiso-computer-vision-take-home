#!/usr/bin/env python3
"""CPU latency with ONNX; includes image load, preprocessing, inference, postprocessing/NMS."""
import argparse,json,os,platform,statistics,subprocess,time
from pathlib import Path
import numpy as np
config_dir = Path(__file__).resolve().parents[1] / ".ultralytics"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
matplotlib_dir = Path(__file__).resolve().parents[1] / ".matplotlib"
matplotlib_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument("--model",default="models/metal_nut_defect.onnx"); p.add_argument("--image",required=True); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--warmup",type=int,default=10); p.add_argument("--runs",type=int,default=100); p.add_argument("--cpu-name",default=None,help="Override processor name if the operating system hides it"); p.add_argument("--out",default="results/generated/cpu_benchmark.json"); a=p.parse_args()
model=YOLO(a.model,task="detect")
for _ in range(a.warmup): model.predict(a.image,imgsz=a.imgsz,device="cpu",verbose=False)
times=[]
for _ in range(a.runs):
    start=time.perf_counter(); model.predict(a.image,imgsz=a.imgsz,device="cpu",verbose=False); times.append((time.perf_counter()-start)*1000)
try: cpu=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip()
except Exception: cpu=platform.processor() or platform.machine()
if a.cpu_name: cpu=a.cpu_name
result={"model":a.model,"cpu":cpu,"platform":platform.platform(),"input_size_px":[a.imgsz,a.imgsz],"warmup_runs":a.warmup,"measured_runs":a.runs,"latency_scope":"image file loading, preprocessing, ONNX CPU inference, and postprocessing/NMS","median_ms":statistics.median(times),"p95_ms":float(np.percentile(times,95)),"mean_ms":statistics.mean(times)}
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
