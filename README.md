# Gauge OCR - Spatial Geometry-Aware Vision Large Model

A deep learning system for pointer gauge reading recognition, combining SAM semantic segmentation, DeepLabV3+ geometric perception, and Qwen2 large language model.

## Project Structure

```
gauge-ocr/
├── README.md
├── requirements.txt
├── setup.py
├── configs/                    # Configuration files
│   ├── default.yaml           # Default configuration
│   ├── train_stage1.yaml      # Stage 1 training configuration
│   ├── train_stage2.yaml      # Stage 2 training configuration
│   └── train_stage3.yaml      # Stage 3 training configuration
├── data/                       # Data module
│   ├── __init__.py
│   ├── datasets.py            # Dataset classes (keypoints, readings)
│   └── transforms.py          # Data augmentation
├── models/                     # Model module
│   ├── __init__.py
│   ├── semantic_branch.py     # SAM + Conv projection
│   ├── geometric_branch.py    # DeepLabV3+ geometric branch (with keypoint head)
│   ├── gate.py                # Gate generator
│   ├── cross_attention.py     # Cross-attention module
│   ├── causal_encoder.py      # DeepEncoder V2 (causal flow + spatial bias)
│   ├── decoder.py             # LLM decoder (Qwen2-0.5B)
│   └── gauge_ocr.py           # Overall model CausalGauge
├── utils/                      # Utility module
│   ├── __init__.py
│   ├── metrics.py             # Evaluation metrics (baseline error, etc.)
│   ├── utils.py               # Helper functions
│   └── visualize.py           # Visualization tools (heatmaps, gate scores)
├── train/                      # Training module
│   ├── __init__.py
│   ├── train_stage1.py        # Stage 1 training script (geometric branch)
│   ├── train_stage2.py        # Stage 2 training script (semantic-geometric fusion, no queries)
│   └── train_stage3.py        # Stage 3 training script (causal flow reordering)
├── scripts/                    # Helper scripts
│   ├── __init__.py
│   ├── prepare_data.py        # Data preprocessing (generate Gaussian heatmaps, etc.)
│   └── evaluate.py            # Evaluation script
└── inference.py                # Inference script
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Training

### Stage 1: Geometric Branch Pre-training
Train the DeepLabV3+ based geometric branch to detect keypoints (pointer tip, center, 4 main scale points).
```bash
python train/train_stage1.py --config configs/train_stage1.yaml
```

**Data Flow in Stage 1:**
- Input: Image [B, 3, 448, 448]
- → Backbone → ASPP → Decoder
- Output: Heatmaps [B, 6, 112, 112] (6 keypoints)
- Loss: MSE + Dice Loss

### Stage 2: Semantic-Geometric Fusion Training
Combine SAM semantic features with geometric features using cross-attention and gating mechanism.
```bash
python train/train_stage2.py --config configs/train_stage2.yaml
```

**Data Flow in Stage 2:**
- Image [B, 3, 448, 448]
  → SAM → Conv projection → Semantic tokens [B, 784, 896]
  → DeepLabV3+ → Geometric features [B, 256, 112, 112]
    → Gate generator → Gate scores [B, 784]
    → Projection → Geometric tokens [B, 784, 896]
  → Cross-attention (semantic as query, geometric as key/value)
  → Causal encoder (no causal queries)
  → LLM Decoder
- Output: Reading text

### Stage 3: Causal Flow Reordering
Introduce learnable causal queries and fine-tune the causal encoder and LLM decoder.
```bash
python train/train_stage3.py --config configs/train_stage3.yaml
```

**Data Flow in Stage 3:**
- Same as Stage 2, but with:
  - Causal queries [B, 784, hidden_size]
  - Concatenated with visual tokens: [B, 1568, hidden_size]
  - Causal mask applied
  - Spatial bias added to attention scores

## Inference

```bash
python inference.py --image_path test.jpg --checkpoint checkpoints/stage3_final.pth
```

## Annotation Formats

### Keypoint Dataset JSON
```json
{
  "image_path": "images/001.jpg",
  "keypoints": [[x0,y0], [x1,y1], ...]  // 6 points, normalized coordinates 0-1
}
```

### Reading Dataset JSON
```json
{
  "image_path": "images/001.jpg",
  "reading": "0.45 MPa"
}
```

## License

MIT License
