import os
import cv2
import numpy as np
import random

def generate_dataset(num_samples=50, img_size=512, output_dir="data"):
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(num_samples):
        # 背景
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # 随机篡改区域
        x = random.randint(50, img_size-150)
        y = random.randint(50, img_size-150)
        w = random.randint(50, 150)
        h = random.randint(50, 150)

        # 篡改类型：粘贴不同颜色块
        tamper_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (x, y), (x+w, y+h), tamper_color, -1)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

        # 保存
        cv2.imwrite(os.path.join(img_dir, f"{i:04d}.jpg"), img)
        cv2.imwrite(os.path.join(mask_dir, f"{i:04d}.png"), mask)

    print(f"生成 {num_samples} 张合成图像及掩码，保存在 data/images 和 data/masks")

if __name__ == "__main__":
    generate_dataset()
