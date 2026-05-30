import torch
import numpy as np 
from Att_UNet.networks.AttUnet import AttU_Net
from PIL import Image
import glob 
import torch.nn as nn 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from Loss_Function import Hybrid_loss
import warnings
import tifffile
warnings.filterwarnings('ignore')

img_url = sorted(glob.glob(r"D:\software\pythonProject\Att_UNet\data_zdx\traindata\imgs/*"))
mask_url = sorted(glob.glob(r"D:\software\pythonProject\Att_UNet\data_zdx\traindata\mask/*"))
# print(img_url)
train_size = int(len(img_url) * 0.8)
train_img_url = img_url[:train_size]
train_mask_url = mask_url[:train_size]
val_img_url = img_url[train_size:]
val_mask_url = mask_url[train_size:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))
epoches = 100
out_channels = 1

def build_model():
    model = AttU_Net(img_ch=3,output_ch=1)
    return model


class LoadDataset(Dataset):  #加载数据集
    def __init__(self, img_url, mask_url,df_used=True,df_norm=True):
        super(LoadDataset, self).__init__()
        self.img_url = img_url
        self.mask_url = mask_url
        self.df_used = df_used
        self.df_norm = df_norm

    def __getitem__(self, idx):
        img = Image.open(self.img_url[idx])
        # img = tifffile.imread(self.img_url[idx])
        # print (img.shape)
        # img = img.resize((256, 256))
        # img = np.resize(img, (256, 256))
        img_array = np.array(img, dtype=np.float32) / 255   #归一化到0-1
        # print(img_array.shape)
        mask = Image.open(self.mask_url[idx])
        # mask = tifffile.imread(self.mask_url[idx])
        #print (mask0.shape)
        #gt1 = np.array(mask0)
        #mask0 = mask0.convert('L')
        # mask = mask.resize((256, 256))
        # mask = np.resize(mask, (256, 256))
        mask = np.array(mask, dtype=np.float32) / 255
        img_array = img_array.transpose(2, 0, 1)    #将波段放在首位
        # img_array = np.transpose(img_array, (2, 0, 1))
        # mask = np.expand_dims(mask, axis=2)         #对mask加一个水体分类波段，与image对应
        # HWC to CHW
        # mask = mask.transpose((2, 0, 1))
        # mask = np.transpose(mask, (2, 0, 1))

        return torch.tensor(img_array.copy()), torch.tensor(mask.copy())

        #return torch.tensor(img_array.copy()), torch.tensor(mask0.copy())


    def __len__(self):
        return len(self.img_url)

def compute_dice(input, target):
    eps = 0.0001
    # input 是经过了sigmoid 之后的输出。Sigmoid 函数是一种常用的激活函数，将输入值映射到一个范围在0到1之间的连续输出
    input = (input > 0.5).float()
    target = (target > 0.5).float()

    # inter = torch.dot(input.view(-1), target.view(-1)) + eps
    inter = torch.sum(target.view(-1) * input.view(-1)) + eps

    # print(self.inter)
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * inter.float()) / union.float()
    return t   #t为计算的 Dice 系数，用于评估二进制分类任务中预测结果与目标结果的相似程度

if __name__ == "__main__":

    model = build_model()
    model.to(device)

    train_dataset = LoadDataset(train_img_url, train_mask_url)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = LoadDataset(val_img_url, val_mask_url)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    loss_func = Hybrid_loss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    #lr = adjust_learning_rate(optimizer, epoch)  # 调整学习率
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.8, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)
    step = 0
    
    for epoch in range(epoches):
        print("epoch is " + str(epoch))
        report_loss = 0.0
        for img, mask in tqdm(train_loader, total=len(train_loader), ncols = 80):
            optimizer.zero_grad()
            step += 1
            img = img.to(device)
            mask = mask.to(device)
            pred_img = model(img) ## pred_img (batch, len, channel, W, H)
            #print (pred_img.shape)
            if out_channels == 1:
                pred_img = pred_img.squeeze(1) # 去掉通道维度
            #loss = loss_func(pred_img, df_out, df_masks, mask0)
            #print (mask0.shape)
            loss = loss_func(pred_img, mask)
            report_loss += loss.item()
            #损失函数的值越小，表示模型的预测结果与真实结果越接近。通过累加每个批次的损失函数值，可以计算出整个训练过程中的总损失函数值，从而了解模型在训练数据上的整体性能
            loss.backward()    #执行反向传播，计算参数的梯度
            optimizer.step()   #根据计算得到的梯度更新模型的参数
            scheduler.step()

            # print("Learning Rate: %.5f." % (scheduler.get_last_lr()))

            # 每一次训练称为一个训练周期（即epoch），一个训练周期内分不同批次进行训练，每个批次为8，所以一个周期内总共训练（样本数/批次）次。
# 一个周期训练完的损失函数=所有训练批次的损失函数累积值，所以report_loss = 0.0每次迭代完loss值都要重置。
            if step % (len(train_loader)) == 0:
                dice = 0.0
                n = 0
                model.eval()
                with torch.no_grad():
                    print("report_loss is " + str(report_loss))   #打印累积损失函数
                    report_loss = 0.0
                    for val_img, val_mask in tqdm(val_loader, total=len(val_loader),ncols = 80):
                        n += 1 #验证批次的数量。通过计算验证数据批次的数量，可以在计算平均 Dice 系数时使用这个数量作为分母，从而得到一个相对准确的平均值
                        val_img = val_img.to(device)
                        val_mask = val_mask.to(device)
                        pred_img = torch.sigmoid(model(val_img))
                        if out_channels == 1:
                            pred_img = pred_img.squeeze(1)
                        cur_dice = compute_dice(pred_img, val_mask)  #计算当前验证批次的 Dice 系数
                        dice += cur_dice.item()  #累加当前验证批次的 Dice 系数
                    dice = dice / n    #平均 Dice 系数
                    print("mean dice is " + str(dice))
                    # scheduler.step(dice)  # 根据平均 Dice 系数调整学习率的调度器。scheduler 是一个学习率调度器对象，根据验证集上的性能指标来动态调整学习率。
                    #if step % len(train_loader) == 0:
                    torch.save(model.state_dict(), "./checkpoints/Epoch-{0}_acc-{1}.pkl".format(epoch+1,  str(round(dice,4))))
                    model.train()

        print("Learning Rate: %.5f." % (scheduler.get_last_lr()))
