"""
ViT分支 独立预测脚本
加载模型: models/vit_best.pth
"""
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def predict(image_path, output_dir):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    orig_size = img.size
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    model = ViTBranch().to(DEVICE)
    model_path = "models/vit_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"[ViT] 加载模型: {model_path}")
    else:
        print(f"[ViT] 错误: 未找到模型文件 {model_path}")
        return

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()

    pred_resized = cv2.resize(pred, orig_size, interpolation=cv2.INTER_LINEAR)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img_output_dir = os.path.join(output_dir, f"{base_name}_ViT")
    os.makedirs(img_output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(img_output_dir, "result.png"), (pred_resized * 255).astype(np.uint8))

    img_array = np.array(img)
    heatmap = cv2.applyColorMap((pred_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(pred_resized, cmap='hot')
    axes[1].set_title(f"ViT Result\nMean: {pred_resized.mean():.3f}")
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(img_output_dir, "visualization.png"), dpi=150, bbox_inches='tight')
    plt.close()

    report = f"""ViT分支检测报告
================
平均置信度: {pred_resized.mean():.4f}
最大置信度: {pred_resized.max():.4f}
最小置信度: {pred_resized.min():.4f}

结论: {'检测到篡改痕迹!' if pred_resized.mean() > 0.5 else '未检测到明显篡改'}
================
"""
    with open(os.path.join(img_output_dir, "报告.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[ViT] 检测完成! 平均置信度: {pred_resized.mean():.4f}")
    print(f"[ViT] 结果保存至: {img_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="ViT分支篡改检测")
    parser.add_argument("image", type=str, help="输入图像路径")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    args = parser.parse_args()

    print(f"[ViT] 使用设备: {DEVICE}")
    predict(args.image, args.output_dir)

if __name__ == "__main__":
    main()
