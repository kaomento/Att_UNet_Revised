# # import numpy
# # import cv2
# # import matplotlib.pyplot as plt
# #
# # image_path = r"C:\Users\user\Desktop\1_color_mask.png"
# # image = cv2.imread(image_path)
# # image.shape
# # plt.imshow(image)
# # plt.show()
# #
# # image_gray = cv2.imread(image_path,0)
# # print ("PIXEL VALUES:\n", "speaker", image_gray[100, 150], "\ncup", image_gray[150, 350])
#
from PIL import Image
import numpy as np

# 打开图像
img = Image.open(r"C:\Users\user\Desktop\1_color_mask.png")
arr = np.array(img)


if arr.shape[2] == 4:
    arr = arr[:, :, :3]

# 展平并找出独特颜色
unique_colors = np.unique(arr.reshape(-1, 3), axis=0)

print(f"图像中有 {len(unique_colors)} 种不同的像素值（颜色）。")
print("独特的 RGB 值：")
for color in unique_colors:
    print(color)


binary_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)

# 定义你的颜色值
color_0 = np.array([70, 70, 70])
color_1 = np.array([128, 64, 128])

# 找到匹配 color_1 的像素位置，并赋值为 1
# (np.all 检查最后一个维度是否三个通道都匹配)
mask_1 = np.all(arr == color_1, axis=-1)
binary_mask[mask_1] = 1

# 此时，匹配 color_0 的位置默认就是 0，无需额外操作
# 如果你想明确赋值，可以这样做：
# mask_0 = np.all(arr == color_0, axis=-1)
# binary_mask[mask_0] = 0

print(f"转换完成！唯一值: {np.unique(binary_mask)}")

label_img = Image.fromarray(binary_mask)
label_img.save(r"C:\Users\user\Desktop\binary_label.png")

# # 2. 导出为可视化的黑白图 (像素值变为 0 和 255)
# # 将 1 乘以 255，让目标变为白色，方便肉眼观察效果
# visual_img = Image.fromarray(binary_mask * 255)
# visual_img.save(r"C:\Users\user\Desktop\binary_visual.png")




# import numpy as np
# from PIL import Image
#
# def get_unique_pixel_values(image_path):
#     # 1. 打开图像
#     img = Image.open(image_path)
#
#     # 2. 转换为 NumPy 数组
#     img_array = np.array(img)
#
#     # 3. 获取所有唯一的像元值
#     # 对于多波段图像，这会返回所有通道中出现过的数字
#     unique_values = np.unique(img_array)
#
#     print(f"图像尺寸: {img.size}")
#     print(f"图像模式: {img.mode}")  # 'L' 为灰度, 'RGB' 为彩色
#     print(f"数据类型: {img_array.dtype}")
#     print(f"唯一的像元值数量: {len(unique_values)}")
#     print(f"具体的像元值: \n{unique_values}")
#
#     return unique_values
#
# # 使用示例
# values = get_unique_pixel_values(r"C:\Users\user\Desktop\binary_label.png")