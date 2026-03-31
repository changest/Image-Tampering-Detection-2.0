import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
import timm
import segmentation_models_pytorch as smp
import cv2

# ================== 配置 ==================
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 数据集 ==================
class TamperDataset(Dataset):
    def __init__(self, data_root, img_size=512):
        self.img_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "masks")
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        base_name = os.path.splitext(img_name)[0]

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, base_name + ".png")).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return img, mask, img_name

# ================== 网络定义（与train_four_branch.py相同）==================
class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
    def forward(self, x):
        return self.model(x)

class ViTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("mit_b2", pretrained=False, features_only=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 1, 3, padding=1)
        )
    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features[-1])

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

class FourBranchFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_branch = CNNBranch()
        self.vit_branch = ViTBranch()
        self.ela_srm_branch = ELASRMBranch()
        self.color_branch = ColorBranch()
        self.fusion_weights = nn.Parameter(torch.ones(4))

    def forward(self, x):
        p1 = self.cnn_branch(x)
        p2 = self.vit_branch(x)
        p3 = self.ela_srm_branch(x)
        p4 = self.color_branch(x)

        target_size = (IMG_SIZE, IMG_SIZE)
        for p in [p1, p2, p3, p4]:
            if p.shape[2:] != target_size:
                p = torch.nn.functional.interpolate(p, size=target_size, mode='bilinear', align_corners=False)

        weights = torch.softmax(self.fusion_weights, dim=0)
        final = weights[0] * p1 + weights[1] * p2 + weights[2] * p3 + weights[3] * p4
        return final, p1, p2, p3, p4, weights

# ================== 评估指标 ==================
def calculate_metrics(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average='binary', zero_division=0
    )
    iou = jaccard_score(target_flat, pred_flat, zero_division=0)

    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}

def evaluate():
    parser = argparse.ArgumentParser(description="评估四分支模型")
    parser.add_argument("--data_root", type=str, default="./data", help="数据根目录")
    parser.add_argument("--model", type=str, default="models/fusion_best.pth", help="融合模型路径")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    args = parser.parse_args()

    print(f"使用设备: {DEVICE}")

    # 加载数据
    dataset = TamperDataset(args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"评估样本数: {len(dataset)}")

    # 加载模型
    model = FourBranchFusionNet().to(DEVICE)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=DEVICE))
        print(f"加载模型: {args.model}")
    else:
        print(f"错误: 未找到模型文件 {args.model}")
        return

    model.eval()

    all_metrics = {'final': [], 'cnn': [], 'vit': [], 'ela_srm': [], 'color': []}

    with torch.no_grad():
        for imgs, masks, _ in tqdm(dataloader, desc="评估中"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            final, p1, p2, p3, p4, _ = model(imgs)

            # 处理每个样本
            for i in range(imgs.shape[0]):
                # 最终融合结果
                pred_final = torch.sigmoid(final[i]) > 0.5
                metrics_final = calculate_metrics(pred_final.cpu().numpy(), masks[i].cpu().numpy())
                all_metrics['final'].append(metrics_final)

                # 各分支结果
                for name, pred in [('cnn', p1[i]), ('vit', p2[i]), ('ela_srm', p3[i]), ('color', p4[i])]:
                    pred_bin = torch.sigmoid(pred) > 0.5
                    metrics = calculate_metrics(pred_bin.cpu().numpy(), masks[i].cpu().numpy())
                    all_metrics[name].append(metrics)

    # 计算平均指标
    def avg_metrics(metrics_list):
        return {
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1': np.mean([m['f1'] for m in metrics_list]),
            'iou': np.mean([m['iou'] for m in metrics_list])
        }

    print("\n========== 四分支模型评估结果 ==========")
    print(f"{'分支':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10}")
    print("-" * 52)

    for branch_name in ['cnn', 'vit', 'ela_srm', 'color', 'final']:
        avg = avg_metrics(all_metrics[branch_name])
        name_map = {'cnn': 'CNN', 'vit': 'ViT', 'ela_srm': 'ELA+SRM', 'color': 'Color', 'final': 'Final Fusion'}
        print(f"{name_map[branch_name]:<12} {avg['precision']:<10.4f} {avg['recall']:<10.4f} {avg['f1']:<10.4f} {avg['iou']:<10.4f}")

    print("=" * 52)

if __name__ == "__main__":
    evaluate()
