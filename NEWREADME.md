
---

# Attention U-Net 遥感影像分割项目 (重制/修复版)

madeby kaomanto 
写在前面我本来只是帮女朋友写她的作业吧，没想过会拿到这样一个傻逼屎山代码，苦于作者没有留下任何信息，我没法去问候作者的母亲，这没法引入原作者的代码仓库。

### ⚠️ 警告：关于原始开源代码的说明

本项目基于某 GitHub 开源项目修复。原作者的代码在逻辑、数据处理和环境兼容性上存在极其严重的**低级错误**。在未经过本项目修复前，该代码**根本无法收敛**，且在非作者本人电脑的环境下完全无法运行。

---

## 🛑 原始代码的“屎山”问题清单

如果你尝试运行原版代码，你会遇到以下让人血压飙升的问题：

1. **路径硬编码 (Hardcoded Paths)**：
原作者直接将 `D:\HuaweiMoveData\...` 这种个人电脑的绝对路径写死在 `glob.glob` 中。
2. **标签处理逻辑错误 (Fatal Label Bug)**：
**这是最离谱的地方。** 原始数据集的 Mask 是 `[0, 1]` 编码，但原作者在 `LoadDataset` 中强行执行了 `mask / 255`。
* **后果**：标签值变成了 `0.0039`。神经网络会认为全图都是背景，导致 `Dice Loss` 永远停留在 **0.99**，模型练到报废也不会有任何效果。


3. **验证逻辑荒谬 (Metric Logic Error)**：
原作者在验证环节的 `dice` 计算中，累加逻辑与平均逻辑错位，甚至导致文件名出现 `acc-1.5`、`acc-2.0` 这种违背数学常识的指标。
4. **维度不匹配 (Shape Mismatch)**：
完全未考虑灰度图 (2D) 与 RGB 图 (3D) 的兼容性，在 `transpose` 换轴时会直接抛出 `ValueError`。
5. **环境崩溃 (NumPy 2.0 Incompatibility)**：
代码未指定依赖版本，在最新的 NumPy 2.x 环境下会因为 PyTorch 编译冲突直接报 `RuntimeError`。
6. **学习率早衰 (LR Scheduler Issue)**：
`StepLR` 设置极其激进，导致模型还没开始收敛，学习率就缩减到了 `0.00000`，训练后期完全是在浪费电费。

---

## ✅ 本版本修复的内容

为了让这个项目达到“人类可运行”的状态，我们进行了以下重构：

### 1. 数据读取层 (Data Layer)

* **强制类型转换**：在读取时加入 `.convert('RGB')` 和 `.convert('L')`，确保输入维度恒定为 `[3, H, W]`。
* **修复归一化**：针对 `[0, 1]` 像素值的 Mask，取消了荒唐的 `/ 255` 操作。
* **统一尺寸**：加入了基于 `PIL` 的 `Resize` 逻辑，并严格区分了原图（双线性）与掩码（最近邻）的插值方式，防止标签污染。

### 2. 核心训练逻辑 (Training Logic)

* **指标重构**：重新编写了 `mean_dice` 的计算逻辑。现在 `Dice` 指标严格限制在 `[0, 1]` 范围内，反映真实的分割精度。
* **学习率优化**：调整了 `scheduler` 的步长，确保模型有足够的训练周期进行收敛。
* **Loss 纠偏**：修正了 `Hybrid_loss` 中由于 Sigmoid 重复调用导致的梯度消失问题。

### 3. 环境适配 (Environment)

* **NumPy 降级建议**：明确要求使用 `numpy<2` 以适配 PyTorch 框架。
* **Headless OpenCV**：测试脚本切换至 `opencv-python-headless`，适配无图形界面的服务器环境。

---

## 🚀 如何运行

### 环境准备

```bash
conda create -n attunet_env python=3.8
conda activate attunet_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" matplotlib tqdm tifffile opencv-python-headless Pillow

```

### 训练

1. 修改 `2_train.py` 中的 `img_url` 和 `mask_url` 为你的实际路径。
2. 运行 `python 2_train.py`。
3. **预期表现**：`Dice Loss` 会在几个 Epoch 内明显下降，`Mean Dice` 应稳步上升至 0.9 以上。

### 测试

1. 创建保存文件夹：`mkdir 222`
2. 修改 `3_test.py` 中的 `pretrained_path` 为生成的 `.pkl` 文件路径。
3. 运行 `python 3_test.py` 查看分割结果。

---

## 💬 结语

写代码开源是为了交流，不是为了通过“屎山”和“Bug”来折磨同行。希望这个修复版能帮到那些被原版代码坑得怀疑人生的开发者。