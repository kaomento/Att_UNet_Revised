from PIL import Image
import numpy as np

# 换成你其中一张 mask 的路径
mask_path = "/root/Att_UNet/dataset/traindata/mask/0_y00_x05.png" 
mask = Image.open(mask_path)
mask_np = np.array(mask)

print(f"唯一像素值: {np.unique(mask_np)}")
print(f"最大值: {mask_np.max()}, 最小值: {mask_np.min()}")