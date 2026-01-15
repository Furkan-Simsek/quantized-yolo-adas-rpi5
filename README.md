# Quantized YOLOv11 and YOLOv8 for Low-Speed ADAS on Raspberry Pi 5

This repository contains the code, models, and benchmarks for the research paper:  
**Real-Time Object Detection with Quantized YOLOv11 and YOLOv8 on Raspberry Pi 5 for Low-Speed ADAS**  
Author: Furkan Şimşek (@FsimsekDev)  
[Download Paper (PDF)](article.pdf) (https://dx.doi.org/10.21203/rs.3.rs-8584571/v1)

## Overview
This study evaluates YOLOv11n/s and YOLOv8n/s models for CPU-only object detection in ADAS on Raspberry Pi 5. Key findings:
- Quantized YOLOv11n achieves ~13 FPS on KITTI with 76.78 ms latency.
- Nano models offer better speed-accuracy trade-off for low-power embedded systems.
- Uses ONNX with INT8 quantization and 30% pruning.

Tested on Raspberry Pi 5 (8GB, performance mode), trained on RTX 3050 Laptop GPU.

## Requirements
- Python 3.8+
- Install dependencies:
  - ultralytics
  - torch
  - onnx
  - onnxruntime
  - opencv-python
