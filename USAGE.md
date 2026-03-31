# Usage Instructions

## Quick Start Methods

### Method 1: Double-click (Recommended)
Double-click the `start.bat` file
- Follow the prompts to enter the image path
- Detection will run automatically

### Method 2: Command Line
Open terminal and navigate to project directory:
```bash
python run_detection.py
```

### Method 3: Direct Execution (for scripts)
```bash
python predict_all.py "path/to/your/image.jpg"
```

## Input Image Path Options

1. **Direct Path Input:**
   ```
   D:\Pictures\test.jpg
   ```

2. **Drag and Drop (Windows):**
   Drag the image file directly into the terminal window, the path will be filled automatically

## Output Locations

Results are saved in the `outputs` folder:
```
outputs/{image_name}_Report/
```

### Included Files:
- `1_CNN.png` - CNN branch result
- `2_ViT.png` - ViT branch result
- `3_ELA_SRM.png` - ELA+SRM branch result
- `4_Color.png` - Color branch result
- `5_Fusion.png` - Fusion result
- `6_Grid.png` - Six-panel comparison
- `7_Overlay.png` - Overlay visualization
- `Report.txt` - Comprehensive report

## Model File Locations

```
models/
  ├── cnn_best.pth      (CNN branch model)
  ├── vit_best.pth      (ViT branch model)
  ├── ela_srm_best.pth  (ELA+SRM branch model)
  └── color_best.pth    (Color branch model)
```

## Notes

1. **Supported formats:** JPG, JPEG, PNG, BMP, TIFF, WEBP
2. **Auto-cleanup:** Old detection records are automatically cleared
3. **GPU required:** Ensure CUDA is installed
4. **Encoding:** Use UTF-8 encoding if you see garbled text

## Troubleshooting

**Q: "Python not detected"**
A: Install Python 3.8+ and add to PATH

**Q: "Model files not found"**
A: Ensure models/ folder contains corresponding .pth files

**Q: Garbled text display**
A: Run in Windows PowerShell: `chcp 65001`
