# Image Tampering Detection System

## Four-Branch Deep Learning Model

This project integrates four deep learning models for image tampering detection, improving accuracy through multi-branch fusion.

## Architecture

| Branch | Model | Features |
|--------|-------|----------|
| CNN | UNet++ (ResNet34) | Local feature extraction, edge detection |
| ViT | Swin Transformer | Global context understanding |
| ELA+SRM | Hand-crafted features fusion | Noise analysis and error level analysis |
| Color | Color anomaly detection | Color consistency and smoothness detection |

## Project Structure

```
fusion_tamper_localization/
├── predict_all.py          # Unified four-branch detection entry
├── train_four_branch.py    # Four-branch joint training
├── augment_data_v2.py      # Data augmentation (8 tampering types)
├── models/                 # Trained model weights
│   ├── cnn_best.pth
│   ├── vit_best.pth
│   ├── ela_srm_best.pth
│   └── color_best.pth
├── start.bat               # Windows one-click launch
├── run_detection.py        # Interactive detection script
└── requirements.txt        # Dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Detection

**Method 1: Double-click (Recommended)**
Double-click `start.bat` file

**Method 2: Command Line**
```bash
python run_detection.py
```

**Method 3: Direct Detection**
```bash
python predict_all.py "path/to/your/image.jpg"
```

## Output Results

Detection results are saved in `outputs/{image_name}_Report/` directory:
- `1_CNN.png` - CNN branch heatmap
- `2_ViT.png` - ViT branch heatmap
- `3_ELA_SRM.png` - ELA+SRM branch result
- `4_Color.png` - Color branch result
- `5_Fusion.png` - Fusion result
- `6_Grid.png` - Six-panel comparison
- `7_Overlay.png` - Overlay visualization
- `Report.txt` - Detection report

## Train Model

### Prepare Data
```bash
python augment_data_v2.py  # Generate synthetic training data
```

### Start Training
```bash
python train_four_branch.py
```

## Detection Thresholds

The system uses graded threshold judgment:
- `> 0.5` - High confidence: Tampering confirmed
- `0.15 - 0.5` - Medium confidence: Possible tampering
- `0.01 - 0.15` - Low confidence: Suspicious tampering
- `< 0.01` - Authentic image

## System Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA
- 8GB+ VRAM (RTX 3060 or higher recommended)
- Windows 10/11 or Linux

## License

MIT License

## Acknowledgments

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [timm](https://github.com/rwightman/pytorch-image-models)
- CASIA Dataset

---

**Author**: changest
**GitHub**: https://github.com/changest/Image-Tampering-Detection-2.0
