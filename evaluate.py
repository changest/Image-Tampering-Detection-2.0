import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
import timm
import segmentation_models_pytorch as smp

# ================== 网络定义（与train.py相同）==================
class SRMConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [
            [[0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0],
             [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0],
             [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1],
             [2, -6, 8, -6, 2],
             [-2, 8, -12, 8, -2],
             [2, -6, 8, -6, 2],
             [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        ]
        filters = torch.tensor(filters, dtype=torch.float32).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * c, 1, h, w)
        x = torch.nn.functional.conv2d(x, self.filters, padding=2)
        x = x.view(b, c * 3, h, w)
        return x

class NoiseStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = SRMConv2d()
        self.encoder = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=9,
            classes=1,
            activation=None
        )

    def forward(self, x):
        x = self.srm(x)
        x = self.encoder(x)
        return x

class RABlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out

class EdgeStream(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = timm.create_model("resnet18", pretrained=False, features_only=True)
        self.encoder = resnet
        self.ra_blocks = nn.ModuleList([
            RABlock(64), RABlock(128), RABlock(256), RABlock(512)
        ])
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
        for i, ra in enumerate(self.ra_blocks):
            if i < len(features):
                features[i] = ra(features[i])
        x = self.decoder(features[-1])
        return x

class FusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, noise_feat, edge_feat, rgb_feat):
        x = torch.cat([noise_feat, edge_feat, rgb_feat], dim=1)
        x = self.fusion(x)
        return x

class TwoStreamFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_stream = NoiseStream()
        self.edge_stream = EdgeStream()
        self.fusion = FusionNetwork()

    def forward(self, x):
        noise_out = self.noise_stream(x)
        edge_out = self.edge_stream(x)
        fused = self.fusion(noise_out, edge_out, x)
        return fused, noise_out, edge_out

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

        return img, mask

# ================== 评估指标 ==================
def calculate_metrics(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average='binary', zero_division=0
    )
    iou = jaccard_score(target_flat, pred_flat, zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }

def evaluate():
    parser = argparse.ArgumentParser(description="评估模型性能")
    parser.add_argument("--data_root", type=str, default="./data", help="数据根目录")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="模型权重路径")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    dataset = TamperDataset(args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"评估样本数: {len(dataset)}")

    # 加载模型
    model = TwoStreamFusionNet().to(device)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"加载模型: {args.model}")
    else:
        print(f"错误: 未找到模型文件 {args.model}")
        return

    model.eval()

    all_metrics = []

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="评估中"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs, _, _ = model(imgs)

            # 阈值处理
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()

            for pred, mask in zip(preds, masks):
                metrics = calculate_metrics(pred.astype(np.uint8), mask.astype(np.uint8))
                all_metrics.append(metrics)

    # 计算平均指标
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics])
    }

    print("\n========== 评估结果 ==========")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall:    {avg_metrics['recall']:.4f}")
    print(f"F1-Score:  {avg_metrics['f1']:.4f}")
    print(f"IoU:       {avg_metrics['iou']:.4f}")
    print("==============================\n")

if __name__ == "__main__":
    import sklearn
    evaluate()
