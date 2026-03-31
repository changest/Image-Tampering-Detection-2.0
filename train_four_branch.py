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
import cv2
from skimage.color import rgb2lab
from scipy.ndimage import uniform_filter

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

        img_tensor = self.img_transform(img)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor, img_name

# ================== 分支1: CNN (UNet++) ==================
class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",  # 使用ImageNet预训练权重
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.model(x)

# ================== 分支2: 轻量级Transformer分支 ==================
class ViTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用轻量级CNN作为基础，模拟Transformer的感受野
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 自定义解码器
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
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ================== 分支3: ELA + SRM (传统特征) ==================
class ELASRMBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # SRM滤波器
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
        self.register_buffer("srm_filters", filters)

        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # 3 SRM + 1 ELA
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def compute_srm(self, x):
        # x: [B, 3, H, W]
        b, c, h, w = x.shape
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x_gray = x_gray.view(b, 1, h, w)
        srm_out = torch.nn.functional.conv2d(x_gray, self.srm_filters, padding=2)
        return srm_out

    def compute_ela(self, x):
        # Error Level Analysis
        b, c, h, w = x.shape
        ela_list = []
        for i in range(b):
            img = x[i].permute(1, 2, 0).cpu().numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img = img.astype(np.uint8)
            # JPEG压缩
            _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            compressed = cv2.imdecode(buf, 1)
            # 计算误差
            ela = np.abs(img.astype(float) - compressed.astype(float))
            ela = np.mean(ela, axis=2, keepdims=True)
            ela = (ela / 255.0).astype(np.float32)
            ela_list.append(torch.from_numpy(ela).permute(2, 0, 1))
        return torch.stack(ela_list).to(x.device)

    def forward(self, x):
        # SRM特征
        srm_feat = self.compute_srm(x)  # [B, 3, H, W]

        # ELA特征
        ela_feat = self.compute_ela(x)  # [B, 1, H, W]

        # 确保尺寸一致
        if srm_feat.shape[2:] != ela_feat.shape[2:]:
            srm_feat = torch.nn.functional.interpolate(
                srm_feat, size=ela_feat.shape[2:], mode='bilinear', align_corners=False
            )

        # 融合
        combined = torch.cat([srm_feat, ela_feat], dim=1)
        out = self.fusion(combined)

        # 上采样到目标尺寸
        if out.shape[2:] != (IMG_SIZE, IMG_SIZE):
            out = torch.nn.functional.interpolate(
                out, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
            )

        return out

# ================== 分支4: 颜色特征 ==================
class ColorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # 颜色异常检测网络
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.entropy_branch = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1)
        )

        self.smoothness_branch = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        # 特征提取
        feat = self.feature_extract(x)

        # 计算局部熵和平滑度
        entropy = self.entropy_branch(feat)
        smoothness = self.smoothness_branch(feat)

        # 结合两者
        out = entropy * torch.sigmoid(smoothness)

        return out

# ================== 四分支融合网络 ==================
class FourBranchFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_branch = CNNBranch()
        self.vit_branch = ViTBranch()
        self.ela_srm_branch = ELASRMBranch()
        self.color_branch = ColorBranch()

        # 自适应融合权重
        self.fusion_weights = nn.Parameter(torch.ones(4))

    def forward(self, x):
        # 四个分支输出
        p1 = self.cnn_branch(x)  # CNN
        p2 = self.vit_branch(x)  # ViT
        p3 = self.ela_srm_branch(x)  # ELA+SRM
        p4 = self.color_branch(x)  # Color

        # 确保所有输出尺寸一致
        target_size = (IMG_SIZE, IMG_SIZE)
        if p1.shape[2:] != target_size:
            p1 = torch.nn.functional.interpolate(p1, size=target_size, mode='bilinear', align_corners=False)
        if p2.shape[2:] != target_size:
            p2 = torch.nn.functional.interpolate(p2, size=target_size, mode='bilinear', align_corners=False)
        if p3.shape[2:] != target_size:
            p3 = torch.nn.functional.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        if p4.shape[2:] != target_size:
            p4 = torch.nn.functional.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)

        # 归一化融合权重
        weights = torch.softmax(self.fusion_weights, dim=0)

        # 加权融合
        final = weights[0] * p1 + weights[1] * p2 + weights[2] * p3 + weights[3] * p4

        return final, p1, p2, p3, p4, weights

# ================== 损失函数 ==================
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# ================== 训练函数 ==================
def train():
    # 数据加载
    train_dataset = TamperDataset(DATA_ROOT, "train", IMG_SIZE)
    val_dataset = TamperDataset(DATA_ROOT, "val", IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 模型
    model = FourBranchFusionNet().to(DEVICE)

    # 优化器 - 为不同分支设置不同学习率
    params = [
        {'params': model.cnn_branch.parameters(), 'lr': LR},
        {'params': model.vit_branch.parameters(), 'lr': LR * 0.5},  # ViT使用较小学习率
        {'params': model.ela_srm_branch.parameters(), 'lr': LR},
        {'params': model.color_branch.parameters(), 'lr': LR},
        {'params': model.fusion_weights, 'lr': LR * 2}  # 融合权重学习率稍大
    ]

    optimizer = optim.AdamW(params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        branch_losses = {'cnn': 0, 'vit': 0, 'ela_srm': 0, 'color': 0, 'fusion': 0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()

            # 混合精度前向
            with torch.cuda.amp.autocast():
                final, p1, p2, p3, p4, weights = model(imgs)

                # 各分支损失
                loss_cnn = nn.functional.binary_cross_entropy_with_logits(p1, masks)
                loss_vit = nn.functional.binary_cross_entropy_with_logits(p2, masks)
                loss_ela = nn.functional.binary_cross_entropy_with_logits(p3, masks)
                loss_color = nn.functional.binary_cross_entropy_with_logits(p4, masks)

                # 融合结果损失
                loss_final = nn.functional.binary_cross_entropy_with_logits(final, masks)
                loss_final += dice_loss(final, masks)

                # 总损失
                loss = loss_final + 0.3 * (loss_cnn + loss_vit + loss_ela + loss_color)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            branch_losses['cnn'] += loss_cnn.item()
            branch_losses['vit'] += loss_vit.item()
            branch_losses['ela_srm'] += loss_ela.item()
            branch_losses['color'] += loss_color.item()
            branch_losses['fusion'] += loss_final.item()

            pbar.set_postfix({
                "total": f"{loss.item():.3f}",
                "fusion": f"{loss_final.item():.3f}"
            })

        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                final, _, _, _, _, weights = model(imgs)
                loss = nn.functional.binary_cross_entropy_with_logits(final, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Fusion Weights: {weights.cpu().detach().numpy()}")

        # 保存各分支模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.cnn_branch.state_dict(), "models/cnn_best.pth")
            torch.save(model.vit_branch.state_dict(), "models/vit_best.pth")
            torch.save(model.ela_srm_branch.state_dict(), "models/ela_srm_best.pth")
            torch.save(model.color_branch.state_dict(), "models/color_best.pth")
            torch.save(model.state_dict(), "models/fusion_best.pth")
            print(f"  保存最佳模型，Val Loss={best_loss:.4f}")

    print("\n训练完成！")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
