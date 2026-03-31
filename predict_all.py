"""
四分支模型统一检测脚本
依次运行四个独立模型，并生成综合报告
"""
import os
import argparse
import subprocess
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import timm
import segmentation_models_pytorch as smp

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 四个分支网络定义 ==================
class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=3, classes=1, activation=None
        )
    def forward(self, x):
        return self.model(x)

class ViTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class ELASRMBranch(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        ]
        filters = torch.tensor(filters, dtype=torch.float32).unsqueeze(1)
        self.register_buffer("srm_filters", filters)
        self.fusion = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def compute_srm(self, x):
        b, c, h, w = x.shape
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x_gray = x_gray.view(b, 1, h, w)
        return torch.nn.functional.conv2d(x_gray, self.srm_filters, padding=2)

    def compute_ela(self, x):
        b, c, h, w = x.shape
        ela_list = []
        for i in range(b):
            img = x[i].permute(1, 2, 0).cpu().numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img = img.astype(np.uint8)
            _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            compressed = cv2.imdecode(buf, 1)
            ela = np.abs(img.astype(float) - compressed.astype(float))
            ela = np.mean(ela, axis=2, keepdims=True) / 255.0
            ela_list.append(torch.from_numpy(ela.astype(np.float32)).permute(2, 0, 1))
        return torch.stack(ela_list).to(x.device)

    def forward(self, x):
        srm_feat = self.compute_srm(x)
        ela_feat = self.compute_ela(x)
        if srm_feat.shape[2:] != ela_feat.shape[2:]:
            srm_feat = torch.nn.functional.interpolate(srm_feat, size=ela_feat.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat([srm_feat, ela_feat], dim=1)
        out = self.fusion(combined)
        if out.shape[2:] != (IMG_SIZE, IMG_SIZE):
            out = torch.nn.functional.interpolate(out, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        return out

class ColorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.entropy_branch = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1)
        )
        self.smoothness_branch = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        feat = self.feature_extract(x)
        entropy = self.entropy_branch(feat)
        smoothness = self.smoothness_branch(feat)
        return entropy * torch.sigmoid(smoothness)

# ================== 统一预测函数 ==================
def predict_all(image_path, output_dir):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    orig_size = img.size
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_array = np.array(img)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img_output_dir = os.path.join(output_dir, f"{base_name}_Report")

    # 清除该图片之前的所有检测记录（包括单独分支和综合检测）
    import shutil
    folders_to_remove = [
        f"{base_name}_CNN",
        f"{base_name}_ViT",
        f"{base_name}_ELA_SRM",
        f"{base_name}_Color",
        f"{base_name}_Report",
        f"{base_name}_检测报告",  # 兼容旧版本
        f"{base_name}_综合检测",  # 兼容旧版本
        base_name  # 可能存在的其他格式
    ]
    removed_count = 0
    for folder in folders_to_remove:
        folder_path = os.path.join(output_dir, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"[INFO] Cleared old detection records: {folder_path}")
            removed_count += 1
    if removed_count == 0:
        print(f"[INFO] No old detection records found")

    os.makedirs(img_output_dir, exist_ok=True)

    results = {}

    # 1. CNN分支
    print("\n[1/4] Running CNN branch...")
    model = CNNBranch().to(DEVICE)
    if os.path.exists("models/cnn_best.pth"):
        model.load_state_dict(torch.load("models/cnn_best.pth", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
    pred_resized = cv2.resize(pred, orig_size, interpolation=cv2.INTER_LINEAR)
    results['CNN'] = pred_resized
    cv2.imwrite(os.path.join(img_output_dir, "1_CNN.png"), (pred_resized * 255).astype(np.uint8))
    print(f"      CNN Confidence: {pred_resized.mean():.4f}")

    # 2. ViT分支
    print("\n[2/4] Running ViT branch...")
    model = ViTBranch().to(DEVICE)
    if os.path.exists("models/vit_best.pth"):
        model.load_state_dict(torch.load("models/vit_best.pth", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
    pred_resized = cv2.resize(pred, orig_size, interpolation=cv2.INTER_LINEAR)
    results['ViT'] = pred_resized
    cv2.imwrite(os.path.join(img_output_dir, "2_ViT.png"), (pred_resized * 255).astype(np.uint8))
    print(f"      ViT Confidence: {pred_resized.mean():.4f}")

    # 3. ELA+SRM分支
    print("\n[3/4] Running ELA+SRM branch...")
    model = ELASRMBranch().to(DEVICE)
    if os.path.exists("models/ela_srm_best.pth"):
        model.load_state_dict(torch.load("models/ela_srm_best.pth", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
    pred_resized = cv2.resize(pred, orig_size, interpolation=cv2.INTER_LINEAR)
    results['ELA_SRM'] = pred_resized
    cv2.imwrite(os.path.join(img_output_dir, "3_ELA_SRM.png"), (pred_resized * 255).astype(np.uint8))
    print(f"      ELA+SRM Confidence: {pred_resized.mean():.4f}")

    # 4. Color分支
    print("\n[4/4] Running Color branch...")
    model = ColorBranch().to(DEVICE)
    if os.path.exists("models/color_best.pth"):
        model.load_state_dict(torch.load("models/color_best.pth", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
    pred_resized = cv2.resize(pred, orig_size, interpolation=cv2.INTER_LINEAR)
    results['Color'] = pred_resized
    cv2.imwrite(os.path.join(img_output_dir, "4_Color.png"), (pred_resized * 255).astype(np.uint8))
    print(f"      Color Confidence: {pred_resized.mean():.4f}")

    # 计算融合结果 (简单平均)
    final = (results['CNN'] + results['ViT'] + results['ELA_SRM'] + results['Color']) / 4
    cv2.imwrite(os.path.join(img_output_dir, "5_Fusion.png"), (final * 255).astype(np.uint8))

    # 生成对比图
    print("\n[OK] Generating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(results['CNN'], cmap='hot')
    axes[0, 1].set_title(f"CNN\nConfidence: {results['CNN'].mean():.3f}")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(results['ViT'], cmap='hot')
    axes[0, 2].set_title(f"ViT\nConfidence: {results['ViT'].mean():.3f}")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(results['ELA_SRM'], cmap='hot')
    axes[1, 0].set_title(f"ELA+SRM\nConfidence: {results['ELA_SRM'].mean():.3f}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(results['Color'], cmap='hot')
    axes[1, 1].set_title(f"Color\nConfidence: {results['Color'].mean():.3f}")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(final, cmap='hot')
    axes[1, 2].set_title(f"Fusion\nConfidence: {final.mean():.3f}")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(img_output_dir, "6_Grid.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 生成叠加图
    heatmap = cv2.applyColorMap((final * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(final, cmap='hot')
    axes[1].set_title("Fusion Heatmap")
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(img_output_dir, "7_Overlay.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 生成综合报告
    report = f"""Image Tampering Detection Report
================================
Image: {base_name}

[Branch Results]
1. CNN Branch
   - Avg Confidence: {results['CNN'].mean():.4f}
   - Max Confidence: {results['CNN'].max():.4f}
   - Status: {'Tampering Detected!' if results['CNN'].mean() > 0.5 else 'No Tampering'}

2. ViT Branch
   - Avg Confidence: {results['ViT'].mean():.4f}
   - Max Confidence: {results['ViT'].max():.4f}
   - Status: {'Tampering Detected!' if results['ViT'].mean() > 0.5 else 'No Tampering'}

3. ELA+SRM Branch
   - Avg Confidence: {results['ELA_SRM'].mean():.4f}
   - Max Confidence: {results['ELA_SRM'].max():.4f}
   - Status: {'Tampering Detected!' if results['ELA_SRM'].mean() > 0.5 else 'No Tampering'}

4. Color Branch
   - Avg Confidence: {results['Color'].mean():.4f}
   - Max Confidence: {results['Color'].max():.4f}
   - Status: {'Tampering Detected!' if results['Color'].mean() > 0.5 else 'No Tampering'}

[Fusion Result]
Avg Confidence: {final.mean():.4f}

[Final Conclusion]
{"=" * 40}
{'⚠️ Image likely tampered!' if final.mean() > 0.5 else '✓ Image likely authentic'}
{"=" * 40}

Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================
"""
    with open(os.path.join(img_output_dir, "Report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # 控制台输出
    print("\n" + "=" * 50)
    print("[Four-branch Detection Complete]")
    print("=" * 50)
    print(f"CNN Confidence:     {results['CNN'].mean():.4f}")
    print(f"ViT Confidence:     {results['ViT'].mean():.4f}")
    print(f"ELA+SRM Confidence: {results['ELA_SRM'].mean():.4f}")
    print(f"Color Confidence:   {results['Color'].mean():.4f}")
    print("-" * 50)
    print(f"Fusion Confidence:  {final.mean():.4f}")
    print("=" * 50)
    # 分级判断 - 阈值根据训练数据调整
    fusion_conf = final.mean()
    if fusion_conf > 0.3:
        conclusion = "HIGH confidence: Image is tampered!"
    elif fusion_conf > 0.01:
        conclusion = "Tampering detected (confidence: {:.2f})".format(fusion_conf)
    else:
        conclusion = "Image appears authentic"

    print(f"Conclusion: {conclusion}")
    print("=" * 50)
    print(f"\nResults saved to: {img_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Four-branch Image Tampering Detection")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    predict_all(args.image, args.output_dir)

if __name__ == "__main__":
    main()
