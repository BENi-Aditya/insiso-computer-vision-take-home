# Metal nut defect detector

Computer Vision Intern take-home for InSiSo Technologies. The repository turns MVTec AD `metal_nut` masks into defect bounding boxes, trains a small YOLO detector, evaluates held-out detections, exports ONNX, and measures CPU latency.

## Current status

The dataset was converted, label overlays reviewed, two models trained, both evaluated on the held-out split, exported to ONNX, and benchmarked on CPU. Metrics and manually reviewed failure cases are recorded in `report/findings.md`. The original archive and derived data are excluded from Git.

## Dataset and use terms

Download MVTec AD from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) using its download form. The page describes over 5,000 high-resolution images across 15 categories with pixel-precise anomaly annotations and licenses the dataset CC BY-NC-SA 4.0; commercial use is not allowed under the published terms. Do not put the dataset in this repository. Confirm with MVTec before any commercial use.

Unpack the archive so this path exists:

```text
data/mvtec_anomaly_detection/metal_nut/
  train/good/
  test/good/
  test/<defect_type>/
  ground_truth/<defect_type>/*_mask.png
```

The preparation script uses normal images from `train/good` and all labeled images in `test/` (including `test/good`). It repartitions images within each normal/defect folder using seed 42 into 70% train, 15% validation, and 15% test. Thus anomalous examples are available for supervised training, unlike the official anomaly-detection protocol. No image is copied to more than one split. This is a small, custom object-detection experiment; its scores are not comparable to official MVTec AD benchmark scores. Small defect folders need at least three examples for a stratified split; the script stops if that condition is not met.

Each connected nonzero region in an anomaly mask becomes one tight axis-aligned box. All defect types share one `defect` class. Normal examples get empty label files. Up to three box overlays per folder are written under `data/metal_nut_yolo/label_audit/`; inspect these before training.

## Setup

Python 3.12.7 was used for the recorded run. Dependencies are pinned for repeatability.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon, the training device is `mps`; use `--device cpu` if MPS is unsupported. Recorded versions are listed in `report/findings.md`. The benchmark is explicitly CPU-only.

## Reproduce

```bash
python scripts/prepare_data.py --raw data/mvtec_anomaly_detection --out data/metal_nut_yolo --seed 42
# Inspect every generated label_audit overlay and the manifest before training.
python scripts/train.py --name baseline_640 --imgsz 640 --epochs 100 --batch 8 --seed 42
python scripts/evaluate.py --weights runs/detect/baseline_640/weights/best.pt --split val --out results/generated/baseline_val
python scripts/evaluate.py --weights runs/detect/baseline_640/weights/best.pt --split test --out results/generated/baseline_test
python scripts/train.py --name experiment_416 --imgsz 416 --epochs 100 --batch 8 --seed 42
python scripts/evaluate.py --weights runs/detect/experiment_416/weights/best.pt --split val --imgsz 416 --out results/generated/experiment_val
python scripts/evaluate.py --weights runs/detect/experiment_416/weights/best.pt --split test --imgsz 416 --out results/generated/experiment_test
python scripts/export_onnx.py --weights runs/detect/baseline_640/weights/best.pt --imgsz 640
python scripts/export_onnx.py --weights runs/detect/experiment_416/weights/best.pt --imgsz 416 --out models/metal_nut_defect_416.onnx
python scripts/benchmark_cpu.py --model models/metal_nut_defect.onnx --image data/metal_nut_yolo/images/test/test__bent__000.png --imgsz 640 --warmup 10 --runs 100 --cpu-name "Apple M5 Max" --out results/generated/baseline_cpu_benchmark.json
python scripts/benchmark_cpu.py --model models/metal_nut_defect_416.onnx --image data/metal_nut_yolo/images/test/test__bent__000.png --imgsz 416 --warmup 10 --runs 100 --cpu-name "Apple M5 Max" --out results/generated/experiment_cpu_benchmark.json
```

The sole planned improvement experiment is input size 640 versus 416, holding checkpoint family, split, seed, epochs, batch, confidence threshold (0.25), and IoU matching (0.5) constant. Choose the experiment using validation results; the test split is for the final comparison only. Use the same test protocol for both variants. Train-time default checkpoint is `yolo11n.pt` (Ultralytics YOLO11 nano pretrained weights); log the resolved library version and checkpoint provenance.

Evaluation reports precision/recall at confidence >=0.25 and greedy one-to-one IoU >=0.5, and single-class 101-point interpolated AP at IoU 0.5, with detections ranked from confidence >=0.001. Read `results/generated/<run>/failure_cases/index.csv`; its associated images show green ground-truth boxes and red predictions. At least five genuine false alarms or misses must be reviewed if they exist. Record the visual evidence and a cautious explanation yourself. If fewer than five exist, state the count and include a threshold analysis without inventing cases.

The CPU benchmark uses the exported ONNX model via Ultralytics' ONNX Runtime integration. It includes image-file loading, preprocessing, CPU inference, and postprocessing/NMS, with 10 warmups and 100 measured runs by default. It records the CPU identifier, input size, median and p95. The CPU name can be passed explicitly when OS access to the processor string is restricted.

## Measured results

| Experiment | Test images / boxes | Precision | Recall | mAP@0.5 | CPU median / p95 |
|---|---:|---:|---:|---:|---:|
| YOLO11n, 640 px | 54 / 25 | 0.576 | 0.760 | 0.770 | 22.75 / 24.20 ms |
| YOLO11n, 416 px | 54 / 25 | 0.630 | 0.680 | 0.791 | 13.75 / 14.40 ms |

Both were trained with seed 42 on MPS and selected by validation. Lowering resolution reduced median end-to-end ONNX CPU latency by about 40% and slightly increased test precision and mAP, while recall fell by 0.08. These figures use the custom split and evaluation method in the report, so they are not official MVTec benchmark scores. CPU: Apple M5 Max; ONNX Runtime 1.23.2 CPUExecutionProvider.

## Deliverables

- `scripts/`: mask conversion, train, evaluate, ONNX export and CPU benchmark.
- `report/findings.md`: submission report, metrics table, failure-case review, A/B/C responses and AI-use disclosure.
- `report/failure_cases/`: five reviewed annotated failure images and dataset attribution.
- `models/metal_nut_defect.onnx` and `models/metal_nut_defect_416.onnx`: exported models included as repository deliverables.
- `results/generated/`: full run outputs; ignored by git by default. Five selected annotated failure examples are included with attribution under `report/failure_cases/`.

## Limitations

This is a small single-class detector trained on a custom partition of a laboratory dataset. MVTec AD is an anomaly-detection dataset, not a factory deployment benchmark. A real line needs production images, an agreed defect definition, calibrated false-alarm/recall targets, capture-condition monitoring, and licensing cleared for the intended use. Tiny defects may be lost at 416 px. A CPU latency measured on this Mac does not predict an edge controller's latency.

## Submission details to complete

- Education: first-year undergraduate student pursuing BITS Pilani CSE (BSc Hons) and CS & AI at Scaler School of Tech; expected graduation in 2030 for both.
- Earliest available start date: 7 January 2027, after returning to Bengaluru from winter break.
- Repository: [BENi-Aditya/insiso-computer-vision-take-home](https://github.com/BENi-Aditya/insiso-computer-vision-take-home) (private; reviewer access has not been shared).
