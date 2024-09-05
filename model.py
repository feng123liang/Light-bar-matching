import math
import warnings
import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
# from torchsummary import summary
import random as r
import copy
from BFdataMaker import *
import time

# 示例用法
from scipy.fftpack import fft, ifft
from scipy import signal

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class Model_with_all_cnn(nn.Module):
    def __init__(self):
        super(Model_with_all_cnn, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=5,kernel_size=9,padding=4),
            nn.BatchNorm2d(5,1e-6),
            nn.ReLU(),
            nn.Conv2d(5,5,kernel_size=2,stride=2),
            nn.Conv2d(in_channels=5,out_channels=10,kernel_size=7,padding = 3),
            nn.BatchNorm2d(10,1e-6),
            nn.ReLU(),
            nn.Conv2d(10,10,kernel_size=2,stride=2),
            nn.Conv2d(in_channels=10,out_channels=15,kernel_size=5,padding = 2),
            nn.BatchNorm2d(15,1e-6),
            nn.ReLU(),
            nn.Conv2d(15,15,kernel_size=2,stride=2),
            nn.Conv2d(in_channels=15,out_channels=30,kernel_size=3,padding = 1),
            nn.BatchNorm2d(30,1e-6),
            nn.ReLU(),
            nn.Conv2d(30,30,kernel_size=2,stride=2),
            nn.Flatten(),
            # nn.Linear(750,750),
            # nn.ReLU(),
            nn.Linear(750,3),
            nn.Softmax(dim = 1)
        )

    def forward(self, input):
        # print(input,input.shape)
        input = input.reshape(-1,1,80,80)   #结果为[128,1,21]  目的是把二维变为三维数据
        x = self.model1(input)
        # print(x.shape)
        
        return x

class Model_concat(nn.Module):
    def __init__(self):
        super(Model_concat, self).__init__()
        self.model_time = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=5,kernel_size=9,padding=4), 
            nn.BatchNorm2d(5,1e-6),
            nn.ReLU(),
            nn.Conv2d(5,5,kernel_size=2,stride=2), # (40,40)
            nn.BatchNorm2d(5,1e-6),
            nn.ReLU(),
        )
        self.model_corp = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=10,kernel_size=7,padding = 3),
            nn.BatchNorm2d(10,1e-6),
            nn.ReLU(),
            nn.Conv2d(10,10,kernel_size=2,stride=2),
            nn.BatchNorm2d(10,1e-6),
            nn.ReLU(),
            nn.Conv2d(in_channels=10,out_channels=15,kernel_size=5,padding = 2),
            nn.BatchNorm2d(15,1e-6),
            nn.ReLU(),
            nn.Conv2d(15,15,kernel_size=2,stride=2),
            nn.BatchNorm2d(15,1e-6),
            nn.ReLU(),
            nn.Conv2d(in_channels=15,out_channels=30,kernel_size=3,padding = 1),
            nn.BatchNorm2d(30,1e-6),
            nn.ReLU(),
            nn.Conv2d(30,30,kernel_size=2,stride=2),
            nn.Flatten(),
            # nn.Linear(750,750),
            # nn.ReLU(),
            nn.Linear(750,3),
            nn.Softmax(dim = 1)
        )
        self.model_fre = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=5,kernel_size=6), 
            nn.BatchNorm2d(5,1e-6),
            nn.ReLU(),
            nn.Conv2d(in_channels=5,out_channels=5,kernel_size=6), 
            nn.BatchNorm2d(5,1e-6),
            nn.ReLU(),
        )
      

    def forward(self, input_tot):
        # print(input,input.shape)
        input_time = input_tot[:,:6400]
        input_fre = input_tot[:,6400:]
        input_time = input_time.reshape(-1,1,80,80)   #结果为[128,1,21]  目的是把二维变为三维数据
        input_time = self.model_time(input_time)

        input_fre = input_fre.reshape(-1,1,50,50)
        input_fre = self.model_fre(input_fre)

        input_corp = torch.cat((input_time,input_fre),dim=1)
        input_corp = self.model_corp(input_corp)
        # print(x.shape)
        
        return input_corp
    

def get_two_loader_from_1D_data(trainX,trainY,valX,valY):
    trainX,valX = torch.FloatTensor(trainX),torch.FloatTensor(valX)

    encoder = LabelEncoder()
    trainY = encoder.fit_transform(trainY.ravel())
    encoder = LabelEncoder()
    valY = encoder.fit_transform(valY.ravel())
    trainY,valY = torch.LongTensor(trainY),torch.LongTensor(valY)

    train_dataset =  torch.utils.data.TensorDataset(trainX, trainY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, )
    test_dataset =  torch.utils.data.TensorDataset(valX, valY)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(valX), shuffle=True, )

    return train_loader,test_loader


def train_model_process(model,train_dataloader,val_dataloader,num_epoches,model_name,output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    # train_loss_all = []
    # val_loss_all = []
    since = time.time()
    
    # with open(output_file, 'w') as f:

    for epoch in range(num_epoches):
        print("Epoch {}/{}".format(epoch + 1, num_epoches))
        print("-" * 10)

        # train_loss = 0.0
        # val_loss = 0.0
        # train_num = 0
        # val_num = 0
        y_true_all = None
        y_pred_all = None
        for step, (bx,by) in enumerate(train_dataloader):

            bx = bx.to(device)
            by = by.to(device)

            model.train()
            output = model(bx)

            # print(output)
            pre_lab = torch.argmax(output,dim = 1)
            loss = criterion(output, by)
            # print(pre_lab)
            if y_true_all == None:
                y_true_all = by
                y_pred_all = pre_lab
            else:
                y_true_all = torch.cat((y_true_all,by))
                y_pred_all = torch.cat((y_pred_all,pre_lab))
            

            
            # print('realy:',by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train_loss += loss.item()* bx.size(0)
            # train_num += bx.size(0)

        # train_loss_all.append(train_loss/train_num)

        evaluate_metrics(y_true_all.cpu(),y_pred_all.cpu(),model_name + "_train.csv",epoch+1)

        y_true_all = None
        y_pred_all = None
        for step, (bx,by) in enumerate(val_dataloader):

            bx = bx.to(device)
            by = by.to(device)

            model.eval()
            output = model(bx)

            pre_lab = torch.argmax(output,dim = 1)
            loss = criterion(output, by)

            if y_true_all == None:
                y_true_all = by
                y_pred_all = pre_lab
            else:
                y_true_all = torch.cat((y_true_all,by))
                y_pred_all = torch.cat((y_pred_all,pre_lab))

            # val_loss += loss.item() * bx.size(0)
            

            # val_num += bx.size(0)

        # train_loss_all.append(train_loss/train_num)
        test_acc = evaluate_metrics(y_true_all.cpu(),y_pred_all.cpu(),model_name + "_test.csv",epoch+1)


        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        if (epoch+1)%100 == 0:
            torch.save(best_model_wts, "pth/" + model_name + "_" + str(epoch+1) + ".pth")
            
        time_use = time.time() - since
        print("Time usage {:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

        
    
    # train_acc_all = train_acc_all.copy()
    # train_loss_all = train_loss_all.copy()
    # val_acc_all = val_acc_all.copy()
    # val_loss_all = val_loss_all.copy()
    # train_process = pd.DataFrame(data = {"epoch": range(num_epoches),
    #                                     "train_acc_all": train_acc_all,
    #                                     "train_loss_all": train_loss_all,
    #                                     "val_acc_all": val_acc_all,
    #                                     "val_loss_all": val_loss_all
    #                                      })

keywordList = ["N09_M07_F10_"]
# datadicts = {"K001":0,"K002":0,'K003':0,"K004":0,"K005":0,"KA04":1,"KA15":1,"KA16":1,"KA22":1,"KA30":1}
datadicts = {"N09_M07_F10_K001":0,"N09_M07_F10_K002":0,'N09_M07_F10_K003':0,"N09_M07_F10_K004":0,"N09_M07_F10_K005":0,
             "N09_M07_F10_KA04":1,"N09_M07_F10_KA15":1,"N09_M07_F10_KA16":1,"N09_M07_F10_KA22":1,"N09_M07_F10_KA30":1,
             "N15_M07_F10_K001": 0,"N15_M01_F10_K001": 0,"N15_M07_F10_K002": 0,"N15_M01_F10_K002": 0,"N15_M07_F10_K003": 0,
             "N15_M01_F10_K003": 0,"N15_M07_F10_K004": 0,"N15_M01_F10_K004": 0,"N15_M07_F10_K005": 0,"N15_M01_F10_K005": 0,
             "N15_M07_F10_KA04": 1,"N15_M01_F10_KA04": 1,"N15_M07_F10_KA15": 1,"N15_M01_F10_KA15": 1,"N15_M07_F10_KA16": 1,
             "N15_M01_F10_KA16": 1,"N15_M07_F10_KA22": 1,"N15_M01_F10_KA22": 1,"N15_M07_F10_KA30": 1,"N15_M01_F10_KA30": 1,
             "N15_M07_F04_K001": 0,"N15_M07_F04_K002": 0,"N15_M07_F04_K003": 0,"N15_M07_F04_K004": 0,"N15_M07_F04_K005": 0,
             "N15_M07_F04_KA04": 1,"N15_M07_F04_KA15": 1,"N15_M07_F04_KA16": 1,"N15_M07_F04_KA22": 1,"N15_M07_F04_KA30": 1,}
            #  "N15_M07_F04_KI04": 2,"N15_M07_F10_KI04": 2,"N15_M01_F10_KI04": 2,"N15_M07_F04_KI14": 2,"N15_M07_F10_KI14": 2,
            #  "N15_M01_F10_KI14": 2,"N15_M07_F04_KI16": 2,"N15_M07_F10_KI16": 2,
            #  "N15_M01_F10_KI16": 2,"N15_M07_F04_KI18": 2,"N15_M07_F10_KI18": 2,"N15_M01_F10_KI18": 2,"N15_M07_F04_KI21": 2,
            #  "N15_M07_F10_KI21": 2,"N15_M01_F10_KI21": 2}

datadicts = {"N15_M07_F04_K001": 0,#"N15_M07_F04_K002": 0,"N15_M07_F04_K003": 0,"N15_M07_F04_K004": 0,"N15_M07_F04_K005": 0,
             "N15_M07_F04_KA04": 1,#"N15_M07_F04_KA15": 1,"N15_M07_F04_KA16": 1,"N15_M07_F04_KA22": 1,"N15_M07_F04_KA30": 1,
             "N15_M07_F04_KI04": 2,}#"N15_M07_F04_KI14": 2,"N15_M07_F04_KI16": 2,"N15_M07_F04_KI18": 2,"N15_M07_F04_KI21": 2}

# datadicts = {"N15_M07_F10_K001": 0,"N15_M01_F10_K001": 0,"N15_M01_F04_K001": 0,"N15_M07_F10_K002": 0,"N15_M01_F10_K002": 0,"N15_M01_F04_K002": 0,"N15_M07_F10_K003": 0,"N15_M01_F10_K003": 0,"N15_M01_F04_K003": 0,"N15_M07_F10_K004": 0,"N15_M01_F10_K004": 0,"N15_M01_F04_K004": 0,"N15_M07_F10_K005": 0,"N15_M01_F10_K005": 0,"N15_M01_F04_K005": 0,"N15_M07_F10_KA04": 1,"N15_M01_F10_KA04": 1,"N15_M01_F04_KA04": 1,"N15_M07_F10_KA15": 1,"N15_M01_F10_KA15": 1,"N15_M01_F04_KA15": 1,"N15_M07_F10_KA16": 1,"N15_M01_F10_KA16": 1,"N15_M01_F04_KA16": 1,"N15_M07_F10_KA22": 1,"N15_M01_F10_KA22": 1,"N15_M01_F04_KA22": 1,"N15_M07_F10_KA30": 1,"N15_M01_F10_KA30": 1,"N15_M01_F04_KA30": 1}
# datadicts = {"K002":0,'K003':0,"K004":0,"K005":0,"KA15":1,"KA16":1,"KA22":1,"KA30":1}
# datadicts = {"N09_M07_F10_K001":0}
# datadicts = {"N09_M07_F10_K001":0,"N15_M07_F04_K001": 0}

# 设置种子
seed = 44
set_seed(seed)

data_folder = "G_lp_td_fd_pixel"
taskname = "test" # folder name as well
output_file = taskname + '.txt'

sample_length = 8901
trainX,trainY,valX,valY = load_data_and_convert_to_grid(datadicts,0.9,data_folder,sample_length)

train_loader,test_loader = get_two_loader_from_1D_data(trainX,trainY,valX,valY)

# # model_with_all_cnn = Model_with_all_cnn()
model_remix = Model_concat()
# # model_state_dict = torch.load('Germanset_processed_data_s64_r3400.pth')

# # 设置模型为评估模式
# # model_with_all_cnn.load_state_dict(model_state_dict)

record = train_model_process(model_remix,train_loader,test_loader,500,taskname,output_file)
