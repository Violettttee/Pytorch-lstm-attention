import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from utils import *
from model import Net


if __name__ == '__main__':
    # 参数设置
    seq_length = 3   # 时间步长
    input_size = 2  # 原本为3，现在为5， 删去postcode与time
    num_layers = 2 # 4
    hidden_size = 256  #128 # 512??
    batch_size = 2
    n_iters = 100000 # 50000 5000
    lr = 2*1e-6     #0.001
    output_size = 1
    split_ratio = 0.9
    path = 'data/raw_sales.csv'
    # path = 'data/ma_lga_12345.csv'
    moudle = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(moudle.parameters(), lr=lr)
    scaler = MinMaxScaler()
    print(moudle)

    df = pd.read_csv(path)
    df['datesold'] = pd.to_datetime(df['datesold'])
    df = df[(df['bedrooms'] >= 2) & (df['bedrooms'] <= 5)]
    # 按照 bedrooms 分组并在季度上重采样
    df = df.groupby('bedrooms').apply(lambda x: x.resample('Q', on='datesold')['price'].median())
    df = df.reset_index(level=0)
    df['datesold'] = df.index        # df.reset_index(inplace=True, drop=False)
    df.reset_index(inplace=True, drop=True)

    data, label = get_Data(path)
    data, label, mm_y, mm_x= normalization(data, label)
    x1,y1 = creat_dataset(data, look_back=seq_length)
    x, y = split_windows(data, seq_length)
    x_data, y_data, x_train, y_train, x_test, y_test = split_data(x, y, split_ratio)
    train_loader, test_loader, num_epochs = data_generator(x_train, y_train, x_test, y_test, n_iters, batch_size)

    iter = 0
    moudle.cuda()
    loss_list = []
    for epochs in range(num_epochs):
        for i, (batch_x, batch_y) in enumerate(train_loader):
            outputs = moudle(batch_x)
            optimizer.zero_grad()  # 将每次传播时的梯度累积清除
            # print(outputs.shape, batch_y.shape)
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()
            iter += 1
            if iter % 100 == 0:
                print("iter: %d, loss: %1.5f" % (iter, loss.item()))
                loss_list.append(loss.item())
            if iter % 45000 == 0:
                plt.plot(loss_list)
                plt.show()
    print(x_data.shape)
    moudle.eval()
    train_predict = moudle(x_data).cpu().detach().numpy() # [:,0] 原本的
    # train_predict = mm_y.inverse_transform(train_predict)[:,0]# tensor (29576,1)
    # median_quarterly_list = plt_price_by_Q_of_DataFrame(df, color=None)
    # df = df.resample('Q', on='datesold')['price'].median()
    # train_predict = mm_y.inverse_transform(train_predict.reshape(-1,1))
    train_predict_df = pd.DataFrame({
        'datesold':list(df['datesold'])[:-(seq_length+1)] ,
        'bedrooms':list(df['bedrooms'])[:-(seq_length+1)] ,
        'price':train_predict ,
    })
    # median_quarterly_list = plt_price_by_Q_of_DataFrame(df, color=None)[:-(seq_length+1)]

    for i in range(2, 6):
        df_bedrooms = df[df['bedrooms'] == i]
        temp = df_bedrooms['datesold']
        plt.plot(df_bedrooms['datesold'].values, df_bedrooms['price'].values, label=f'Bedrooms: {i}')

    for i in range(2, 6):
        df_bedrooms = train_predict_df[train_predict_df['bedrooms'] == i]
        plt.plot(df_bedrooms['datesold'].values, mm_y.inverse_transform(df_bedrooms['price'].values.reshape(-1,1)), color='black')

    plt.legend()
    plt.show()

