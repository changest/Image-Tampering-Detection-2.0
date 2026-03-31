import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import timm
import segmentation_models_pytorch as smp

# ================== 配置参数 ==================
DATA_ROOT = "./data"
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 数据加载 ==================
class TamperDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=512):
        self.img_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "masks")
        self.img_size = img_size

        all_imgs = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])

        # 简单划分：80%训练，20%验证
        split_idx = int(len(all_imgs) * 0.8)
        if split == "train":
            self.img_list = all_imgs[:split_idx]
        else:
            self.img_list = all_imgs[split_idx:]

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

# ================== 噪声流：SRM滤波 ==================
class SRMConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        # SRM高通滤波器核
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
        # x: [B, 3, H, W]
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
            in_channels=9,  # 3 channels * 3 filters
            classes=1,
            activation=None
        )

    def forward(self, x):
        x = self.srm(x)
        x = self.encoder(x)
        return x

# ================== 边缘流：RA模块 ==================
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
        # 使用ResNet18作为骨干
        resnet = timm.create_model("resnet18", pretrained=False, features_only=True)
        self.encoder = resnet

        # RA模块 - 处理不同通道数
        self.ra_blocks = nn.ModuleList([
            RABlock(64),
            RABlock(64),   # layer1
            RABlock(128),  # layer2
            RABlock(256),  # layer3
            RABlock(512)   # layer4
        ])

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        features = self.encoder(x)

        # 应用RA模块
        for i, ra in enumerate(self.ra_blocks):
            if i < len(features):
                features[i] = ra(features[i])

        # 使用最后一层特征
        x = self.decoder(features[-1])
        return x

# ================== 融合网络 ==================
class FusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # noise(1) + edge(1) + rgb(3) = 5 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, noise_feat, edge_feat, rgb_feat):
        # 确保所有特征图尺寸一致
        target_size = noise_feat.shape[2:]
        if edge_feat.shape[2:] != target_size:
            edge_feat = torch.nn.functional.interpolate(
                edge_feat, size=target_size, mode='bilinear', align_corners=False
            )
        if rgb_feat.shape[2:] != target_size:
            rgb_feat = torch.nn.functional.interpolate(
                rgb_feat, size=target_size, mode='bilinear', align_corners=False
            )
        # 特征融合
        x = torch.cat([noise_feat, edge_feat, rgb_feat], dim=1)
        x = self.fusion(x)
        return x

# ================== 完整双流网络 ==================
class TwoStreamFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_stream = NoiseStream()
        self.edge_stream = EdgeStream()
        self.fusion = FusionNetwork()

    def forward(self, x):
        noise_out = self.noise_stream(x)
        edge_out = self.edge_stream(x)

        # 融合
        fused = self.fusion(noise_out, edge_out, x)

        return fused, noise_out, edge_out

# ================== 训练函数 ==================
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train():
    # 数据加载
    train_dataset = TamperDataset(DATA_ROOT, "train", IMG_SIZE)
    val_dataset = TamperDataset(DATA_ROOT, "val", IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 模型
    model = TwoStreamFusionNet().to(DEVICE)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练循环
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs, noise_out, edge_out = model(imgs)

            # BCE + Dice 损失
            bce_loss = nn.functional.binary_cross_entropy_with_logits(outputs, masks)
            d_loss = dice_loss(outputs, masks)
            loss = bce_loss + d_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs, _, _ = model(imgs)
                loss = nn.functional.binary_cross_entropy_with_logits(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"保存最佳模型，Val Loss={best_loss:.4f}")

    print("训练完成！")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
