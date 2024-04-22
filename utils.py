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

# 文件读取
def get_Data(data_path):
    type_mapping = {'house': 1, 'unit': 0}

    # data=pd.read_excel(data_path)
    data=pd.read_csv(data_path)
    # data=data.iloc[:,:3]  # 以三个特征作为数据
    # label=data.iloc[:,2:] # 取最后一个特征作为标签
    # data=data[data['bedrooms'] == 3]
    data = data[(data['bedrooms'] >= 2) & (data['bedrooms'] <= 5)]
    # data=data.iloc[:,]
    data['propertyType'] = data['propertyType'].map(type_mapping)
    data['datesold'] = pd.to_datetime(data['datesold'])
    # data = data[(data['datesold'] >='2013') & (data['datesold'] <='2018')]
    # data['datesold'] = (data['datesold']-pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    # data=data[['datesold', 'postcode', 'propertyType', 'bedrooms', 'price']]
    # data=data[['datesold', 'propertyType', 'bedrooms', 'price']]
    # data=data[['propertyType', 'bedrooms', 'price']]
    # data=data[['bedrooms', 'price']]
    # M->梯度爆炸问题？？？？？？？？
    median_quarterly = data.groupby('bedrooms').apply(lambda x: x.resample('Q', on='datesold')['price'].median())
    median_quarterly_with_bedrooms = median_quarterly.reset_index(level=0)
    median_quarterly_with_bedrooms['bedrooms'] = median_quarterly_with_bedrooms.index

    # median_quarterly = data.groupby('bedrooms').resample('M', on='datesold')['price'].median()
    # data=data[['price']]

    # label=data.iloc[:,-1:] # price列

    data = median_quarterly.to_frame()
    data.reset_index(inplace=True)
    data=data[['bedrooms','price']]
    #     data.columns.values[0] = 'price'
    data.columns.values[0] = 'bedrooms'
    label=data.iloc[:,-1:] # price列
    print(data.head())
    print(label.head())
    return data,label

# 数据预处理
def normalization(data,label):

    mm_x=MinMaxScaler() # 导入sklearn的预处理容器
    mm_y=MinMaxScaler()
    data=data.values    # 将pd的系列格式转换为np的数组格式
    label=label.values
    data=mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
    label=mm_y.fit_transform(label)
    # 顺便返回原数据 ndarray类型，还是换成pd.DataFrame吧
    origin_data=mm_x.inverse_transform(data)
    return data,label,mm_y,mm_x

def creat_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:i + look_back]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 时间向量转换
def split_windows(data,seq_length):
    x=[]
    y=[]
    for i in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _x=data[i:(i+seq_length),:]    # 最后4个不会有
        _y=data[i+seq_length,-1]     # 少的是最后4个
        x.append(_x)
        y.append(_y)
    x,y=np.array(x),np.array(y)
    print('x.shape,y.shape=\n',x.shape,y.shape)
    return x,y

# 数据分离
def split_data(x,y,split_ratio):

    train_size=int(len(y)*split_ratio)
    test_size=len(y)-train_size

    x_data=Variable(torch.Tensor(np.array(x)).to('cuda'))
    y_data=Variable(torch.Tensor(np.array(y)).to('cuda'))

    x_train=Variable(torch.Tensor(np.array(x[0:train_size])).to('cuda'))
    y_train=Variable(torch.Tensor(np.array(y[0:train_size])).to('cuda'))
    y_test=Variable(torch.Tensor(np.array(y[train_size:len(y)])).to('cuda'))
    x_test=Variable(torch.Tensor(np.array(x[train_size:len(x)])).to('cuda'))

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
    .format(x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape))

    return x_data,y_data,x_train,y_train,x_test,y_test

# 数据装入
def data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size):

    num_epochs=n_iters/(len(x_train)/batch_size) # n_iters代表一次迭代
    num_epochs=int(num_epochs)
    train_dataset=Data.TensorDataset(x_train,y_train)
    test_dataset=Data.TensorDataset(x_train,y_train)
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,drop_last=True) # 加载数据集,使数据集可迭代
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)

    return train_loader,test_loader,num_epochs


def plt_price_by_Q_of_DataFrame(origin_x_data, color):
    results = {}
    median_quarterly_list = []
    for bedroom in origin_x_data['bedrooms'].unique():
        # 筛选出当前卧室数量的数据
        subset = origin_x_data[origin_x_data['bedrooms'] == bedroom]
        # 按季度对价格进行聚合计算，并获取中位数
        median_quarterly = subset.resample('Q', on='datesold')['price'].median()
        # 将结果存储在字典中
        results[bedroom] = median_quarterly
    for bedroom in range(2, 6):
        if bedroom in results:
            # 获取当前卧室数量对应的季度中位数数据
            median_quarterly = results[bedroom]
            temp = median_quarterly.values.reshape(-1, 1)
            # median_quarterly_list = median_quarterly.index.to_numpy()
            median_quarterly_list.append(median_quarterly.index.to_numpy())
            # mm_x.fit_transform(temp) ->y轴值
            # 绘制折线图
            if color != None:
                plt.plot(median_quarterly.index.to_numpy(), median_quarterly.values, label=f'Bedrooms: {bedroom}',
                         color=color)
            else:
                plt.plot(median_quarterly.index.to_numpy(), mm_x.fit_transform(temp), label=f'Bedrooms: {bedroom}')
    return median_quarterly_list