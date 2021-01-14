import dgl
import torch
from mlload import get_ml_100k,get_fshl
from torch.utils.data import TensorDataset, DataLoader
from minibatch import MinibatchSampler
from gcmc import rmse, GCMCRating
import tqdm
import numpy as np
import pandas as pd
# pd.set_option('display.width', 1000)  # 设置字符显示宽度
# pd.set_option('display.max_rows', None)  # 设置显示最大行
# np.set_printoptions(threshold=np.inf)

# 从初始输入里拿到数据
# train_data, test_data, user_data, item_data = get_ml_100k()
train_data, user_data, item_data = get_fshl()

# 把用户id和项目id的类型换成枚举类
train_data = train_data.astype({'user_id': 'category', 'item_id': 'category'})
# test_data = test_data.astype({'user_id': 'category', 'item_id': 'category'})

print("在保持一致操作之前")
print(train_data.head(5), '\n')
print(train_data.tail(5), '\n')
# print(test_data.head(5), '\n')
# print(test_data.tail(5), '\n')
print("train_item: \n", train_data.item_id)
# print("test_item: \n", test_data.item_id)
print("以上\n")

# 训练集和测试集的数据保持一致
# 两个集合里用户数量都是943，id一致
# 训练集有1680个item，而测试集只有1127，所以这一步的关键就是把测试集的item_id枚举类型类别的总数改成了1680
# 通过对比，确认了正是这一步操作，导致后续测试集运行时产生了数组下标越界问题
# test_data['user_id'].cat.set_categories(train_data['user_id'].cat.categories, inplace=True)
# test_data['item_id'].cat.set_categories(train_data['item_id'].cat.categories, inplace=True)

print("在保持一致操作之后")
print(train_data.head(5), '\n')
print(train_data.tail(5), '\n')
# print(test_data.head(5), '\n')
# print(test_data.tail(5), '\n')
print("train_item: \n", train_data.item_id)
# print("test_item: \n", test_data.item_id)
print("以上\n")

# 这两步操作从训练集和测试集数据里提取出0开始的相对位置ID
# 用于后续数据集的创建，以及异构图graph的创建
train_user_ids = torch.LongTensor(train_data['user_id'].cat.codes.values)
train_item_ids = torch.LongTensor(train_data['item_id'].cat.codes.values)
train_data['rating'] = train_data['rating'].astype(float)
train_ratings = torch.LongTensor(train_data['rating'].values)

# test_user_ids = torch.LongTensor(test_data['user_id'].cat.codes.values)
# test_item_ids = torch.LongTensor(test_data['item_id'].cat.codes.values)
# test_ratings = torch.LongTensor(test_data['rating'].values)

print("看看基于cat.codes的ID：\n")
print(train_data['item_id'], '\n')
# print(test_data['item_id'], '\n')
print(train_item_ids, '\n')
# print(test_item_ids, '\n')

# 创建异构图
graph = dgl.heterograph({
    ('user', 'watched', 'item'): (train_user_ids, train_item_ids),
    ('item', 'watchedby', 'user'): (train_item_ids, train_user_ids)
})

print(graph)
print(graph.etypes)

# 令训练数据和用户、项目数据一致
print("处理前：\n")
print("item_data: \n", item_data)
user_data[0] = user_data[0].astype('category')
user_data[0] = user_data[0].cat.set_categories(train_data['user_id'].cat.categories)
# 更新了类别后，训练数据集的类别可能比整体用户数的类别少，导致产生空值
# 为此需要进行空值去除，具体就是把第0列的空值去掉
user_data = user_data.dropna(subset=[0])
# codes是类别数值到索引值的映射，把原来很长的用户id映射成从0开始的位置整数
user_data[0] = user_data[0].cat.codes
user_data = user_data.sort_values(0)

# 训练集的项目数量有变化，整体数据为1682个，但训练集里只有1680个
# 用户数都是943，没变化
# 所以这里操作的含义就是缩减项目数，让第0列的index数值符合实际的item数量
item_data[0] = item_data[0].astype('category')
item_data[0] = item_data[0].cat.set_categories(train_data['item_id'].cat.categories)
item_data = item_data.dropna(subset=[0])
item_data[0] = item_data[0].cat.codes
item_data = item_data.sort_values(0)

print("处理后：\n")
print("user_data: \n", user_data)
print("item_data: \n", item_data)


# 处理用户的年龄、性别、职业，以及项目的体裁one-hot向量
user_data[1] = user_data[1].astype('category')
# user_data[3] = user_data[3].astype('category')
# user_data[4] = user_data[4].astype('category')

# user_age = user_data[1].values // 10
# num_user_age_bins = user_age.max() + 1

user_gender = user_data[1].cat.codes.values
num_user_genders = len(user_data[1].cat.categories)

# user_occupation = user_data[3].cat.codes.values
# num_user_occupations = len(user_data[3].cat.categories)

item_data[1] = item_data[1].astype('category')
# item_genres = item_data[1].values
# num_item_genres = item_data.drop_duplicates(subset=[1],keep='last')
# num_item_genres = len(num_item_genres)
# print(num_item_genres)
# num_item_genres = item_genres.shape[0]


item_genres = item_data[1].cat.codes.values
num_item_genres = len(item_data[1].cat.categories)


# 将上述特征赋予图中的结点
# graph.nodes['user'].data['age'] = torch.LongTensor(user_age)
graph.nodes['user'].data['gender'] = torch.LongTensor(user_gender)
# graph.nodes['user'].data['occupation'] = torch.LongTensor(user_occupation)

graph.nodes['item'].data['genres'] = torch.LongTensor(item_genres)

# 本来直接用边类型就可以，但是会报错，只好用全称
graph.edges[('item', 'watchedby', 'user')].data['rating'] = torch.LongTensor(train_ratings)
graph.edges[('user', 'watched', 'item')].data['rating'] = torch.LongTensor(train_ratings)

# 设置数据集
train_dataset = TensorDataset(train_user_ids, train_item_ids, train_ratings)
# test_dataset = TensorDataset(test_user_ids, test_item_ids, test_ratings)

# 定义训练过程
NUM_LAYERS = 1
BATCH_SIZE = 500
NUM_EPOCHS = 50

HIDDEN_DIMS = 8
sampler = MinibatchSampler(graph, NUM_LAYERS)

# 准备加载数据
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=sampler.sample, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=sampler.sample, shuffle=False)

# 创建模型
# model = GCMCRating(graph.number_of_nodes('user'), graph.number_of_nodes('item'), HIDDEN_DIMS, 5, NUM_LAYERS,
#                    num_user_age_bins, num_user_genders, num_user_occupations, num_item_genres)
model = GCMCRating(graph.number_of_nodes('user'), graph.number_of_nodes('item'), HIDDEN_DIMS, 5, NUM_LAYERS,
                    num_user_genders, num_item_genres)
print('node',graph.number_of_nodes('user'),graph.number_of_nodes('item'))
# 使用Adam优化器
opt = torch.optim.Adam(model.parameters())

# 开始训练
for _ in range(NUM_EPOCHS):
    model.train()
    # 加个进度条，直观
    # 首先是训练
    with tqdm.tqdm(train_dataloader) as t:
        predictions = []
        ratings = []
        for pair_graph, blocks in t:
            user_emb, item_emb = model(blocks)
            prediction = model.compute_score(pair_graph, user_emb, item_emb)
            loss = ((prediction - pair_graph.edata['rating']) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            t.set_postfix({'loss': '%.4f' % loss.item()}, refresh=False)
            ratings.append(pair_graph.edata['rating'])
            predictions.append(prediction)
        predictions = torch.cat(predictions, 0)
        ratings = torch.cat(ratings, 0)
    model.eval()
    # 其次是测试
    # with tqdm.tqdm(test_dataloader) as t:
    #     with torch.no_grad():
    #         predictions = []
    #         ratings = []
    #         for pair_graph, blocks in t:
    #             user_emb, item_emb = model(blocks)
    #             prediction = model.compute_score(pair_graph, user_emb, item_emb)
    #             predictions.append(prediction)
    #             ratings.append(pair_graph.edata['rating'])
    #         predictions = torch.cat(predictions, 0)
    #         ratings = torch.cat(ratings, 0)
    print('RMSE:', rmse(predictions, ratings).item())


