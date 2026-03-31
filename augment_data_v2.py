"""
增强版数据生成脚本 - 更真实的篡改类型
使用方法: python augment_data_v2.py
"""
import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def apply_random_transform(img, mask=None):
    """随机变换"""
    h, w = img.shape[:2]

    # 随机旋转
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 随机翻转
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        if mask is not None:
            mask = cv2.flip(mask, 1)
    if random.random() > 0.5:
        img = cv2.flip(img, 0)
        if mask is not None:
            mask = cv2.flip(mask, 0)

    # 随机缩放
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        if mask is not None:
            mask = cv2.resize(mask, (new_w, new_h))
        # 裁剪或填充到原尺寸
        if scale > 1:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            img = img[start_y:start_y+h, start_x:start_x+w]
            if mask is not None:
                mask = mask[start_y:start_y+h, start_x:start_x+w]
        else:
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
            img = cv2.resize(img, (w, h))
            if mask is not None:
                mask = cv2.copyMakeBorder(mask, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT)
                mask = cv2.resize(mask, (w, h))

    # 随机亮度/对比度
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if mask is not None:
        return img, mask
    return img

def poisson_blend(src, dst, mask, center):
    """泊松融合 - 更自然的拼接"""
    try:
        # 确保mask是单通道
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 归一化mask
        mask = (mask > 0).astype(np.uint8) * 255

        # 调整src大小以匹配mask
        x, y = center
        h, w = mask.shape
        src_resized = cv2.resize(src, (w, h))

        # 使用seamlessClone进行泊松融合
        result = cv2.seamlessClone(src_resized, dst, mask, center, cv2.NORMAL_CLONE)
        return result
    except:
        # 如果失败，使用简单融合
        return simple_blend(src, dst, mask, center)

def simple_blend(src, dst, mask, center):
    """简单alpha融合"""
    x, y = center
    h, w = dst.shape[:2]
    src_h, src_w = src.shape[:2]

    # 计算位置
    x1 = max(0, x - src_w//2)
    y1 = max(0, y - src_h//2)
    x2 = min(w, x1 + src_w)
    y2 = min(h, y1 + src_h)

    # 调整src大小
    src_resized = cv2.resize(src, (x2-x1, y2-y1))

    # 创建mask
    mask_region = np.zeros((y2-y1, x2-x1), dtype=np.float32)
    cv2.circle(mask_region, ((x2-x1)//2, (y2-y1)//2), min(x2-x1, y2-y1)//2, 1.0, -1)
    mask_region = gaussian_filter(mask_region, sigma=5)

    # 融合
    mask_3ch = np.stack([mask_region]*3, axis=-1)
    dst[y1:y2, x1:x2] = (dst[y1:y2, x1:x2] * (1 - mask_3ch) + src_resized * mask_3ch).astype(np.uint8)

    # 返回mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = (mask_region * 255).astype(np.uint8)
    return dst, full_mask

def copy_move_advanced(img):
    """高级复制移动 - 带旋转缩放的复制"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 选择源区域（避免边缘）
    src_w = random.randint(w//6, w//3)
    src_h = random.randint(h//6, h//3)
    src_x = random.randint(20, w - src_w - 20)
    src_y = random.randint(20, h - src_h - 20)

    # 提取区域
    region = img[src_y:src_y+src_h, src_x:src_x+src_w].copy()

    # 随机变换
    angle = random.uniform(-30, 30)
    scale = random.uniform(0.7, 1.3)

    # 旋转变换
    center = (src_w // 2, src_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    transformed = cv2.warpAffine(region, M, (src_w, src_h), borderMode=cv2.BORDER_REFLECT)

    # 选择目标位置
    dst_x = random.randint(20, w - src_w - 20)
    dst_y = random.randint(20, h - src_h - 20)

    # 如果重叠太多，重新选择
    if abs(dst_x - src_x) < src_w and abs(dst_y - src_y) < src_h:
        return copy_move_simple(img)

    # 使用泊松融合
    tampered, mask_region = simple_blend(transformed, img.copy(), None, (dst_x + src_w//2, dst_y + src_h//2))
    mask = mask_region

    return tampered, mask

def copy_move_simple(img):
    """简单复制移动"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    block_w = random.randint(w//8, w//4)
    block_h = random.randint(h//8, h//4)
    src_x = random.randint(0, w - block_w - 1)
    src_y = random.randint(0, h - block_h - 1)
    dst_x = random.randint(0, w - block_w - 1)
    dst_y = random.randint(0, h - block_h - 1)

    # 避免重叠
    if abs(dst_x - src_x) < block_w//2 and abs(dst_y - src_y) < block_h//2:
        dst_x = (dst_x + w//2) % (w - block_w)
        dst_y = (dst_y + h//2) % (h - block_h)

    block = img[src_y:src_y+block_h, src_x:src_x+block_w].copy()

    # 添加轻微噪声和模糊使篡改更真实
    if random.random() > 0.5:
        block = cv2.GaussianBlur(block, (3, 3), 0.5)

    tampered = img.copy()
    tampered[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = block
    mask[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = 255

    return tampered, mask

def splicing_realistic(img):
    """真实感拼接 - 从同一图片其他区域取样"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    tampered = img.copy()

    num_blocks = random.randint(1, 3)

    for _ in range(num_blocks):
        # 随机形状（椭圆或不规则多边形）
        shape_type = random.choice(['ellipse', 'rect', 'poly'])

        if shape_type == 'ellipse':
            # 椭圆区域
            cx = random.randint(w//4, 3*w//4)
            cy = random.randint(h//4, 3*h//4)
            axes = (random.randint(30, 80), random.randint(30, 80))
            angle = random.randint(0, 180)

            # 从其他位置取样
            src_cx = random.randint(50, w-50)
            src_cy = random.randint(50, h-50)

            # 创建mask
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(temp_mask, (cx, cy), axes, angle, 0, 360, 255, -1)

            # 复制区域
            region_h, region_w = axes[1]*2, axes[0]*2
            if src_cy + region_h < h and src_cx + region_w < w:
                region = img[src_cy:src_cy+region_h, src_cx:src_cx+region_w]
                region = cv2.resize(region, (region_w, region_h))

                # 应用
                tampered[temp_mask > 0] = region[:np.sum(temp_mask > 0)]
                mask = cv2.bitwise_or(mask, temp_mask)

        elif shape_type == 'rect':
            # 矩形区域带羽化边缘
            bw = random.randint(50, 150)
            bh = random.randint(50, 150)
            x = random.randint(20, w - bw - 20)
            y = random.randint(20, h - bh - 20)

            src_x = random.randint(0, w - bw)
            src_y = random.randint(0, h - bh)

            region = img[src_y:src_y+bh, src_x:src_x+bw].copy()

            # 添加后处理
            if random.random() > 0.5:
                # 轻微模糊
                region = cv2.GaussianBlur(region, (3, 3), 0.5)
            if random.random() > 0.5:
                # 调整亮度
                region = cv2.convertScaleAbs(region, alpha=random.uniform(0.9, 1.1), beta=0)

            tampered[y:y+bh, x:x+bw] = region

            # 羽化边缘
            roi = mask[y:y+bh, x:x+bw]
            cv2.rectangle(roi, (0, 0), (bw, bh), 255, -1)
            roi = cv2.GaussianBlur(roi, (15, 15), 0)
            mask[y:y+bh, x:x+bw] = roi

        else:  # poly
            # 不规则多边形
            pts = []
            cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
            for _ in range(random.randint(4, 8)):
                px = cx + random.randint(-40, 40)
                py = cy + random.randint(-40, 40)
                pts.append([px, py])
            pts = np.array(pts, np.int32)

            cv2.fillPoly(mask, [pts], 255)

            # 从其他位置取样填充
            src_x = random.randint(0, w - 80)
            src_y = random.randint(0, h - 80)
            region = img[src_y:src_y+80, src_x:src_x+80]

            # 调整大小匹配多边形区域
            poly_mask = np.zeros_like(mask)
            cv2.fillPoly(poly_mask, [pts], 255)
            ys, xs = np.where(poly_mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                region_resized = cv2.resize(region, (max(xs)-min(xs)+1, max(ys)-min(ys)+1))
                tampered[poly_mask > 0] = region_resized.flatten()[:len(ys)]

    return tampered, mask

def inpainting_tampering(img):
    """擦除修复篡改"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    num_holes = random.randint(1, 3)
    tampered = img.copy()

    for _ in range(num_holes):
        # 随机形状
        shape = random.choice(['circle', 'rect', 'ellipse'])

        if shape == 'circle':
            radius = random.randint(15, 50)
            x = random.randint(radius, w - radius)
            y = random.randint(radius, h - radius)
            cv2.circle(mask, (x, y), radius, 255, -1)

        elif shape == 'rect':
            rw = random.randint(30, 80)
            rh = random.randint(30, 80)
            x = random.randint(0, w - rw)
            y = random.randint(0, h - rh)
            cv2.rectangle(mask, (x, y), (x+rw, y+rh), 255, -1)

        else:  # ellipse
            cx = random.randint(50, w-50)
            cy = random.randint(50, h-50)
            axes = (random.randint(20, 50), random.randint(20, 50))
            angle = random.randint(0, 180)
            cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

    # 使用OpenCV修复
    inpaint_radius = random.randint(3, 7)
    tampered = cv2.inpaint(tampered, mask, inpaint_radius, cv2.INPAINT_TELEA)

    return tampered, mask

def jpeg_compression_artifact(img):
    """JPEG压缩伪影篡改 - 不同质量系数拼接"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 将图片分成两部分，用不同质量压缩
    split_pos = random.randint(w//3, 2*w//3)

    # 左半部分低质量
    left = img[:, :split_pos]
    _, buf = cv2.imencode('.jpg', left, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(50, 70)])
    left_low = cv2.imdecode(buf, 1)

    # 右半部分高质量
    right = img[:, split_pos:]
    _, buf = cv2.imencode('.jpg', right, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(90, 95)])
    right_high = cv2.imdecode(buf, 1)

    # 合并
    tampered = np.hstack([left_low, right_high])

    # 标记边界区域为篡改区域
    cv2.line(mask, (split_pos-10, 0), (split_pos-10, h), 255, 20)

    return tampered, mask

def noise_inconsistency(img):
    """噪声不一致篡改"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 选择区域添加不同噪声
    num_regions = random.randint(1, 3)
    tampered = img.copy()

    for _ in range(num_regions):
        x = random.randint(0, w-100)
        y = random.randint(0, h-100)
        rw = random.randint(50, 100)
        rh = random.randint(50, 100)

        # 添加高斯噪声
        noise = np.random.normal(0, random.randint(10, 25), (rh, rw, 3))
        tampered[y:y+rh, x:x+rw] = np.clip(tampered[y:y+rh, x:x+rw] + noise, 0, 255).astype(np.uint8)

        mask[y:y+rh, x:x+rw] = 255

    return tampered, mask

def blur_sharpen_tampering(img):
    """模糊锐化篡改"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    tampered = img.copy()

    num_regions = random.randint(1, 2)

    for _ in range(num_regions):
        x = random.randint(0, w-100)
        y = random.randint(0, h-100)
        rw = random.randint(60, 120)
        rh = random.randint(60, 120)

        region = tampered[y:y+rh, x:x+rw]

        if random.random() > 0.5:
            # 模糊
            region = cv2.GaussianBlur(region, (5, 5), random.uniform(1, 3))
        else:
            # 锐化
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            region = cv2.filter2D(region, -1, kernel)

        tampered[y:y+rh, x:x+rw] = region
        mask[y:y+rh, x:x+rw] = 255

    return tampered, mask

def color_adjustment_tampering(img):
    """色彩调整篡改"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    num_regions = random.randint(1, 2)
    tampered = img.copy()

    for _ in range(num_regions):
        x = random.randint(0, w-80)
        y = random.randint(0, h-80)
        rw = random.randint(50, 100)
        rh = random.randint(50, 100)

        region = tampered[y:y+rh, x:x+rw].astype(np.float32)

        # 色彩调整
        adjustment = random.choice(['hue', 'saturation', 'brightness', 'contrast'])

        if adjustment == 'hue':
            # 色调调整
            hsv = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(10, 30)) % 180
            region = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        elif adjustment == 'saturation':
            hsv = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.5, 1.5), 0, 255)
            region = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        elif adjustment == 'brightness':
            region = np.clip(region + random.randint(-30, 30), 0, 255).astype(np.uint8)

        else:  # contrast
            region = np.clip(region * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)

        tampered[y:y+rh, x:x+rw] = region
        mask[y:y+rh, x:x+rw] = 255

    return tampered, mask

def generate_tampering_v2(img):
    """生成篡改样本 - 增强版"""
    tamper_types = [
        copy_move_simple,
        copy_move_advanced,
        splicing_realistic,
        inpainting_tampering,
        jpeg_compression_artifact,
        noise_inconsistency,
        blur_sharpen_tampering,
        color_adjustment_tampering,
    ]

    # 随机选择一种篡改类型
    tamper_func = random.choice(tamper_types)

    try:
        tampered, mask = tamper_func(img)
    except Exception as e:
        # 如果失败，使用简单复制移动
        tampered, mask = copy_move_simple(img)

    return tampered, mask

def augment_dataset_v2(input_dir="data/images", output_dir="data", target_count=1000):
    """
    扩增数据集 - 增强版
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
    print(f"[INFO] Tampering types: Copy-Move, Splicing, Inpainting, JPEG artifacts, Noise, Blur/Sharpen, Color adjustment")

    for i in tqdm(range(target_count)):
        # 随机选择一张基础图片
        base_img = random.choice(original_images).copy()

        # 应用随机变换（先变换原图）
        base_img = apply_random_transform(base_img)

        # 60%概率生成篡改样本，40%保持正常
        if random.random() > 0.4:
            img, mask = generate_tampering_v2(base_img)
        else:
            img = base_img
            mask = np.zeros(base_img.shape[:2], dtype=np.uint8)

        # 保存
        cv2.imwrite(os.path.join(img_dir, f"{i:04d}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(os.path.join(mask_dir, f"{i:04d}.png"), mask)

    print(f"[OK] Generated {target_count} samples in {output_dir}")
    print(f"    - Images: {img_dir}")
    print(f"    - Masks: {mask_dir}")
    print(f"[INFO] Sample distribution: ~60% tampered, ~40% authentic")

if __name__ == "__main__":
    # 生成1000张训练样本
    augment_dataset_v2(target_count=1000)
