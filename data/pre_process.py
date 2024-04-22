import pandas as pd
import matplotlib.pyplot as plt

'''
原始数据绘图结果
'''

df = pd.read_csv("raw_sales.csv")

df['datesold'] = pd.to_datetime(df['datesold'])
results = {}
for bedroom in df['bedrooms'].unique():
    # 筛选出当前卧室数量的数据
    subset = df[df['bedrooms'] == bedroom]
    # 按季度对价格进行聚合计算，并获取中位数
    median_quarterly = subset.resample('M', on='datesold')['price'].median()
    # 将结果存储在字典中
    results[bedroom] = median_quarterly
# 显示结果
for bedroom, median_quarterly in results.items():
    print(f"Bedrooms: {bedroom}")
    print(median_quarterly)

# 使用 groupby 方法按照卧室数量进行分组，并按季度对价格进行聚合计算，并获取中位数
# results = df.groupby('bedrooms').resample('Q', on='datesold')['price'].median()

# 获取卧室数量的具体值
# bedrooms_values = list(results.groupby(level=0).groups.keys())
# 创建 Matplotlib 图形对象
plt.figure(figsize=(10, 6))

# 遍历 results 字典，绘制折线图
for bedroom in range(1, 5):
    if bedroom in results:
        # 获取当前卧室数量对应的季度中位数数据
        median_quarterly = results[bedroom]
        # 绘制折线图
        plt.plot(median_quarterly.index.to_numpy(), median_quarterly.values, label=f'Bedrooms: {bedroom}')
# 添加标题和标签
plt.title('Median Price Monthly by Bedrooms')
plt.xlabel('Monthly')
plt.ylabel('Median Price')
plt.legend()  # 添加图例

# 显示图形
plt.show()
exit()
df.set_index('datesold', inplace=True)
# 计算 'price' 列在每个季度的中位数
price_median_quarterly_by_bedrooms = df.groupby('bedrooms')['price'].resample('Q').median()
price_median_quarterly = df['price'].resample('Q').median()
plt.plot(price_median_quarterly.index.to_numpy(), price_median_quarterly.values)
# 添加标题和标签
plt.title('Median Price Quarterly')
plt.xlabel('Quarter')
plt.ylabel('Median Price')

# 显示图形
plt.show()
exit()

# df['year'] = df['datesold'].dt.year
# df['quarter'] = df['datesold'].dt.quarter
# df['year_quarter'] = df['year'].astype(str) + '-' + df['quarter'].astype(str)

bedrooms_dic = {}
time_dic = {}


for index,row in df.iterrows():
    line_data = list(row)
    bedrooms = row[4]
    # print(type(bedrooms)) int
    price = row[2]
    time = row[5]
    # print(type(time)) str
    if bedrooms not in bedrooms_dic:
        bedrooms_dic[bedrooms] = [price]
        time_dic[bedrooms] = [time]
    else:
        bedrooms_dic[bedrooms].append(price)
        time_dic[bedrooms].append(time)
    # print(line_data)
    # exit()

plt.plot(time_dic[3], bedrooms_dic[3])
# 添加标题和标签
plt.title('Sample Line Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
# print(bedrooms_dic)
# print(data)