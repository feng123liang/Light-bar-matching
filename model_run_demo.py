

import torch
import time
from model import *
import numpy as np
import pandas as pd
import datetime
from client import *

# 格式化输出当前时间
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')



groups = ["pump_100_BRB_H_filtered_cated/Electric_Motor-2_100_time-broken rotor bar-ch1.csv",
          "pump_100_BRB_H_filtered_cated/Electric_Motor-2_100_time-broken rotor bar-ch2.csv",
          "pump_100_BRB_H_filtered_cated/Electric_Motor-2_100_time-broken rotor bar-ch3.csv",
          "pump_100_BRB_H_filtered_cated\Electric_Motor-2_100_time-broken rotor bar-ch4.csv",
          "pump_100_BRB_H_filtered_cated\Electric_Motor-2_100_time-broken rotor bar-ch5.csv",
          "pump_100_BRB_H_filtered_cated\Electric_Motor-2_100_time-broken rotor bar-ch6.csv"
          ]

datas = pd.read_csv(groups[0]).values[:,1]
for i in range(1,6):
    datas = np.vstack((datas,pd.read_csv(groups[i]).values[:,1]))

model_name = 'test_200'
model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model_state_dict = torch.load('pth/' + model_name + '.pth')
model.load_state_dict(model_state_dict)

mqtt_ = MQTTClient()
mqtt_.run()

index = 0
highindex = (datas.shape[1])
print(highindex)

wholeTime = time.time()
predict_dict = {0:'healthy',1:'broken rotor bar'}
sample_indices = np.arange(0, 6400, 16) # 6400/16 = 400

for i in range(100):
    since = time.time()
    
    if(index + 6400 >= highindex):
        sub = index + 6400 - highindex
        tem = datas[0,index:].copy() 
        tem = np.hstack((tem,datas[0,:sub].copy()))
        transfer = datas[:,index:]
        transfer = np.hstack((transfer,datas[:,:sub]))
        print(transfer.shape)
        index = sub
    else:
        tem = datas[0,index:index+6400].copy()
        transfer = datas[:,index:index+6400]
        index += 6400
    # print(index, tem.shape)
    tem = (tem - min(tem)) / (max(tem) - min(tem)) * 255
    tem = torch.FloatTensor(tem.reshape(80,80))
    print(apply_model(model,tem))

    mqtt_.my_publish(0,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     predict_dict[0],transfer[:,sample_indices].tolist(),None)
    
    time_use = time.time() - since
    print(time_use)

wholeTime = time.time() - wholeTime
print("in total:", wholeTime)