1. 如果数据集特别大，例如几千x几千，需自行裁剪小一点，例如512x512，否则训练容易显存不足。
2. 将掩膜mask做拉伸，二分类的话就是将目标物体像素记为1，背景记为2，该步骤的运行代码即为1_read.py,
   运行完此代码后即将dataset/traindata/mask1里的文件转换为dataset/traindata/mask里的文件，看似是全黑图像，实则有值，
   可以放入Arcgis里面进行查看。
3. 运行2_train.py即可做训练
4. 运行3_test.py即可做预测和精度评估

注：整个代码流程如上，仅需将相应文件夹里面填取对应数据集即可，数据集是放在dataset文件夹下，分为traindata和testdata。

