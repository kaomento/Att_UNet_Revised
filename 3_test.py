from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import torch.nn.functional as F
# from sklearn.metrics import accuracy_score
#from networks.UNetFormer import UNetFormer
from networks.AttUnet import AttU_Net
from utils.dataset import BasicDataset
from PIL import Image

pretrained_path = r'/root/Att_UNet/checkpoints/Epoch-20_dice-0.9827.pkl'
#scales = [0.5, 1.0, 1.5]
scales = [1.0]
OA_all = []
P_all = []
R_all = []
F1_all = []
Miou_all = []
classes = 1
normMean = [0.46099097, 0.32533738, 0.32106236]
normStd = [0.20980413, 0.1538582, 0.1491854]
crop_h = 256
crop_w = 256

# 构建模型，载入训练好的权重参数
net = AttU_Net(img_ch=3,output_ch=1)
net.eval()

if torch.cuda.is_available():
    #支持cuda计算的情况下
    net = net.cuda()
    pretrained_dict = torch.load(pretrained_path,map_location=torch.device('cuda'))
    net.load_state_dict(pretrained_dict)
else:
    # 不支持cuda计算的情况下
    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    net.load_state_dict(pretrained_dict)

dir_img = r"/root/Att_UNet/dataset/testdata/imgs/"
dir_mask = r"/root/Att_UNet/dataset/testdata/mask/"
dataset = BasicDataset(dir_img, dir_mask, 1)

test_loader = DataLoader(dataset, batch_size = 1, shuffle = False)

def net_process(model, image, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if torch.cuda.is_available():
        input = input.unsqueeze(0).cuda()
    else:
        input = input.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.sigmoid(output)
    output = output.squeeze(0)

    output = output.data.cpu().numpy()
    #print (output1.shape)
    #output1 = (output1>0.5)
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, stride_rate=2/3):
    #print (image.shape)
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    #print (image.shape)
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model,image_crop)
    #print (prediction_crop.shape)
    prediction_crop /= np.expand_dims(count_crop, 2)
    #print (prediction_crop.shape)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    #prediction_crop = cv2.cvtColor(prediction_crop,cv2.COLOR_BGR2RGB)
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    #prediction = prediction_crop.resize(w,h)
    #print (prediction.shape)
    prediction = np.expand_dims(prediction, 2)
    return prediction
 
# 获得混淆矩阵
def BinaryConfusionMatrix(prediction, groundtruth):
    """Computes scores:
    TP = True Positives    真正例
    FP = False Positives   假正例
    FN = False Negatives   假负例
    TN = True Negatives    真负例
    return: TP, FP, FN, TN"""
 
    # TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    # FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    # FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    # TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    # np.float已经被弃用了,将代码中的np.float替换为float
    TP = float(np.sum((prediction == 1) & (groundtruth == 1)))
    FP = float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = float(np.sum((prediction == 0) & (groundtruth == 1)))
    TN = float(np.sum((prediction == 0) & (groundtruth == 0)))
 
    return TN, FP, FN,TP
 
# 精准率和 或 查准率的计算方法
def get_precision(prediction, groundtruth):
    _, FP, _, TP = BinaryConfusionMatrix(prediction, groundtruth)
    precision = float(TP) / (float(TP+FP)+ 1e-6)
    return precision
 
# 召回率和 或 查全率的计算方法
def get_recall(prediction, groundtruth):
    TN, FP, FN,TP = BinaryConfusionMatrix(prediction, groundtruth)
    recall =  float(TP)/(float(TP+FN) + 1e-6)
    return recall
 
# 准确率的计算方法
def get_accuracy(prediction, groundtruth):
    TN, FP, FN,TP = BinaryConfusionMatrix(prediction, groundtruth)
    accuracy = float(TP+TN)/(float(TP + FP + FN + TN) + 1e-6)
    return accuracy
 
def get_sensitivity(prediction, groundtruth):
    return get_recall(prediction, groundtruth)
 
def get_specificity(prediction, groundtruth):
    TN, FP, FN,TP = BinaryConfusionMatrix(prediction, groundtruth)
    specificity = float(TN)/(float(TN+FP) + 1e-6) 
    return specificity
 
def get_f1_score(prediction, groundtruth):
    precision = get_precision(prediction, groundtruth)
    recall =  get_recall(prediction, groundtruth)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score
 
# Dice相似度系数，计算两个样本的相似度，取值范围为[0, 1], 分割结果最好为1，最坏为0
def get_dice(prediction, groundtruth):
    TN, FP, FN,TP = BinaryConfusionMatrix(prediction, groundtruth)    
    dice = 2 * float(TP)/(float(FP + 2 * TP + FN) + 1e-6)
    return dice
 
#交并比 一般都是基于类进行计算, 值为1这一类的iou
def get_iou1(prediction, groundtruth):
    TN, FP, FN,TP = BinaryConfusionMatrix(prediction, groundtruth)
    iou = float(TP)/(float(FP + TP + FN) + 1e-6)
    return iou
 
#交并比 一般都是基于类进行计算, 值为0这一类的iou
def get_iou0(prediction, groundtruth):
    TN, FP, FN,TP = BinaryConfusionMatrix(prediction, groundtruth)
    iou = float(TN)/(float(FP + TN + FN) + 1e-6)
    return iou
# 基于类进行计算的IoU就是将每一类的IoU计算之后累加，再进行平均，得到的就是基于全局的评价
# 平均交并比
def get_mean_iou(prediction, groundtruth):
    iou0 = get_iou0(prediction, groundtruth)
    iou1 = get_iou1(prediction, groundtruth)
    mean_iou = (iou1 + iou0)/2
    return mean_iou

if __name__ == '__main__':
    # 确保保存路径存在
    import os
    if not os.path.exists("./222/"):
        os.makedirs("./222/")

    for i, batch in enumerate(test_loader):
        print(f"正在处理第 {i+1} 张图片...")
        input, label = batch['image'], batch['mask0']
        
        # input 从 loader 出来是 [1, 3, H, W]
        input = np.squeeze(input.numpy(), axis=0) # 变成 [3, H, W]
        image = np.transpose(input, (1, 2, 0))    # 变成 [H, W, 3]
        
        # 【核心修复】确保 image 即使是灰度图也被视为 3 通道
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.merge([image, image, image])

        h, w, _ = image.shape # 现在这里绝对不会报错了
        
        prediction = np.zeros((h, w, classes), dtype=float)
        # ... 后续代码保持不变 ...
        for scale in scales:
            base_size = 0
            if h > w:
                base_size = h
            else:
                base_size = w
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            #print (image_scale)
            prediction += scale_process(net, image_scale, classes, crop_h, crop_w, h, w, normMean)
        prediction0 = prediction/len(scales)
        prediction0 = (prediction0>0.4)
        prediction = prediction0*255.0
        cv2.imwrite("./222/"+ str(i+1)+ '.png',prediction)
    #     prediction1 = prediction0.flatten()
    #     #print (prediction1)
    #     #label = rgb2gray(label)
    #     #label = np.where(label >= 0.05, 1, 0)
    #     label1 = label.numpy().flatten()
    #     #TN, FP, FN, TP = BinaryConfusionMatrix(prediction1, label1)
    #     OA = get_accuracy(prediction1, label1)
    #     p = get_precision(prediction1, label1)
    #     r = get_recall(prediction1, label1)
    #     f1 = get_f1_score(prediction1, label1)
    #     miou = get_mean_iou(prediction1, label1)
    #     #prediction = np.argmax(prediction, axis=2)
    #     #color_annotation(prediction, output_path + str(i) + ".png")
    #     #OA = accuracy_score(np.reshape(prediction,[-1]), np.reshape(label,[-1]))
    #     print("the " + str(i+1) + "th image's OA:" + str(OA))
    #     print("the " + str(i+1) + "th image's P:" + str(p))
    #     print("the " + str(i+1) + "th image's R:" + str(r))
    #     print("the " + str(i+1) + "th image's F1:" + str(f1))
    #     print("the " + str(i+1) + "th image's Miou:" + str(miou))
    #     OA_all.append(OA)
    #     P_all.append(p)
    #     R_all.append(r)
    #     F1_all.append(f1)
    #     Miou_all.append(miou)
    # print("OA:",np.mean(OA_all))
    # print("P:",np.mean(P_all))
    # print("R:",np.mean(R_all))
    # print("F1:",np.mean(F1_all))
    # print("Miou:",np.mean(Miou_all))











