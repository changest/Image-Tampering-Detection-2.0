"""
数据增强脚本 - 基于现有图片生成更多训练样本
使用方法: python augment_data.py
"""
import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def apply_random_transform(img):
    """随机变换: 旋转、翻转、缩放、亮度调整"""
    h, w = img.shape[:2]

    # 随机旋转
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 随机翻转
    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # 水平翻转
    if random.random() > 0.5:
        img = cv2.flip(img, 0)  # 垂直翻转

    # 随机亮度/对比度
    alpha = random.uniform(0.8, 1.2)  # 对比度
    beta = random.randint(-20, 20)    # 亮度
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img

def generate_tampering(img):
    """生成篡改样本"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    tampered = img.copy()

    # 选择篡改类型
    tamper_type = random.choice(['copy_move', 'splicing', 'inpainting'])

    if tamper_type == 'copy_move':
        # 复制移动: 复制一个区域粘贴到另一个位置
        block_w = random.randint(w//8, w//4)
        block_h = random.randint(h//8, h//4)
        src_x = random.randint(0, w - block_w - 1)
        src_y = random.randint(0, h - block_h - 1)
        dst_x = random.randint(0, w - block_w - 1)
        dst_y = random.randint(0, h - block_h - 1)

        # 复制区域
        block = tampered[src_y:src_y+block_h, src_x:src_x+block_w].copy()
        # 粘贴到目标位置
        tampered[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = block
        # 更新mask
        mask[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = 255

    elif tamper_type == 'splicing':
        # 拼接: 添加不同颜色的矩形块
        num_blocks = random.randint(1, 3)
        for _ in range(num_blocks):
            bw = random.randint(w//10, w//5)
            bh = random.randint(h//10, h//5)
            x = random.randint(0, w - bw)
            y = random.randint(0, h - bh)

            # 随机颜色块或从其他区域复制
            if random.random() > 0.5:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(tampered, (x, y), (x+bw, y+bh), color, -1)
            else:
                src_x = random.randint(0, w - bw)
                src_y = random.randint(0, h - bh)
                tampered[y:y+bh, x:x+bw] = tampered[src_y:src_y+bh, src_x:src_x+bw]

            cv2.rectangle(mask, (x, y), (x+bw, y+bh), 255, -1)

    else:  # inpainting
        # 擦除修复: 用周围像素填充
        num_holes = random.randint(1, 3)
        for _ in range(num_holes):
            radius = random.randint(20, 50)
            x = random.randint(radius, w - radius)
            y = random.randint(radius, h - radius)

            # 创建圆形掩码
            cv2.circle(mask, (x, y), radius, 255, -1)

        # 使用OpenCV修复
        tampered = cv2.inpaint(tampered, mask, 3, cv2.INPAINT_TELEA)

    return tampered, mask

def augment_dataset(input_dir="data/images", output_dir="data", target_count=500):
    """
    扩增数据集
    input_dir: 原始图片目录
    output_dir: 输出目录
    target_count: 目标样本数
    """
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 读取原始图片
    original_images = []
    if os.path.exists(input_dir):
        for fname in sorted(os.listdir(input_dir)):
            if fname.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(input_dir, fname))
                if img is not None:
                    original_images.append(img)

    if len(original_images) == 0:
        print("[ERROR] No images found in", input_dir)
        return

    print(f"[INFO] Found {len(original_images)} original images")
    print(f"[INFO] Generating {target_count} augmented samples...")

    for i in tqdm(range(target_count)):
        # 随机选择一张基础图片
        base_img = random.choice(original_images).copy()

        # 应用随机变换
        base_img = apply_random_transform(base_img)

        # 50%概率生成篡改样本，50%保持正常
        if random.random() > 0.5:
            img, mask = generate_tampering(base_img)
        else:
            img = base_img
            mask = np.zeros(base_img.shape[:2], dtype=np.uint8)

        # 保存
        cv2.imwrite(os.path.join(img_dir, f"{i:04d}.jpg"), img)
        cv2.imwrite(os.path.join(mask_dir, f"{i:04d}.png"), mask)

    print(f"[OK] Generated {target_count} samples in {output_dir}")
    print(f"    - Images: {img_dir}")
    print(f"    - Masks: {mask_dir}")

if __name__ == "__main__":
    # 生成500张训练样本
    augment_dataset(target_count=500)
