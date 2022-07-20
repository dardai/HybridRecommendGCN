import dgl
import torch
from DGL.mlload import get_ml_100k, get_fshl, get_bigraph
from torch.utils.data import TensorDataset, DataLoader
from DGL.minibatch import MinibatchSampler
from DGL.gcmc import rmse, GCMCRating1, GCMCRating2
import tqdm
import pandas as pd
import numpy as np

FSLflag = False


# 计算分类的记录字典
def makeClassifyDict(item_data, FSLflag):
    print("make classify dict")
    classifydict = {}
    item = item_data
    item = item.values.tolist()
    for row in item:
        if row[0] not in classifydict.keys():
            classifydict[row[0]] = []
            if FSLflag == False:
                for i in range(5, 24):
                    if row[i] == 1:
                        classifydict[row[0]].append(i)
            else:
                classifydict[row[0]].append(row[1])
        else:
            continue
    return classifydict


# 在计算分类的覆盖情况
def recommend(item_data, topK, FSLflag, classify_num=0):
    if FSLflag == False:
        result = pd.read_csv('file_saved/ml-DGLresult.csv')
    else:
        result = pd.read_csv('file_saved/fsl-DGLresult.csv')
    result_topK = result.groupby(['user_id']).head(topK).reset_index(drop=True)
    # 计算列表覆盖率
    recommend_length = len(result_topK['item_id'].value_counts())
    item_length = len(item_data)
    cov = recommend_length / item_length
    print("the rate of coverage: ")
    print(cov)
    # 计算物品类别覆盖率
    # 记录推荐类别的字典
    classify_num_dict = {}
    classify = makeClassifyDict(item_data, FSLflag)
    if FSLflag:
        item_id = result_topK['item_id'].astype('str').values.tolist()
        classify_id = []
        for i in item_id:
            classify_id.append(classify[i][0])
    else:
        item_id = result_topK['item_id'].value_counts().keys().values.tolist()
    if FSLflag == False:
        for row in item_id:
            for c in classify[row]:
                if isinstance(c, int):
                    if c not in classify_num_dict.keys():
                        classify_num_dict[c] = 1
                    else:
                        continue
                else:
                    for i in c:
                        if i not in classify_num_dict.keys():
                            classify_num_dict[i] = 1
                        else:
                            continue
    else:
        classify_num_dict = set(classify_id)
    if FSLflag == False:
        classify_cov = (len(classify_num_dict) * 1.0) / 19.0
    else:
        classify_num = classify_num
        classify_cov = (len(classify_num_dict) * 1.0) / classify_num
    print("the rate of classify coverage: ")
    print(classify_cov)


def dglMainFSL(layers, batch_size, epochs, hiddeen_dims, topK):
    # 从初始输入里拿到数据
    train_data, user_data, item_data, classify_data = get_bigraph()
    train_data = train_data.astype({'user_id': 'str', 'item_id': 'str'})
    item_data_for_recommend = item_data.copy()
    rating_count = train_data['rating'].value_counts().values.tolist()
    NUM_RATINGS = len(rating_count)

    classify_id = classify_data['classify_id'].values.tolist()
    classify_num = len(set(classify_id))

    # 把用户id和项目id的类型换成枚举类
    train_data = train_data.astype({'user_id': 'category', 'item_id': 'category'})

    # 训练集和测试集的数据保持一致

    # 这两步操作从训练集和测试集数据里提取出0开始的相对位置ID
    # 用于后续数据集的创建，以及异构图graph的创建
    train_user_ids = torch.LongTensor(train_data['user_id'].cat.codes.values)
    train_item_ids = torch.LongTensor(train_data['item_id'].cat.codes.values)
    train_data['rating'] = train_data['rating'].astype(float)
    train_ratings = torch.LongTensor(train_data['rating'].values)

    # 全连接图的建立，准备user的id，item的id，item的id只留训练过的
    all_user_ids = list(set(train_user_ids.tolist()))
    all_item_ids = list(set(train_item_ids.tolist()))
    all_user_ids = [val for val in all_user_ids for i in range(len(all_item_ids))]
    all_item_ids = all_item_ids * len(set(all_user_ids))

    # 建立字典对应前后id
    user_dict = dict(zip(train_user_ids.tolist(), train_data['user_id'].values.tolist()))
    item_dict = dict(zip(train_item_ids.tolist(), train_data['item_id'].values.tolist()))

    # 创建异构图
    graph = dgl.heterograph({
        ('user', 'watched', 'item'): (train_user_ids, train_item_ids),
        ('item', 'watchedby', 'user'): (train_item_ids, train_user_ids)
    })
    # 令训练数据和用户、项目数据一致
    user_data[0] = user_data[0].astype('category')
    user_data[0] = user_data[0].cat.set_categories(train_data['user_id'].astype('category').cat.categories)
    # 更新了类别后，训练数据集的类别可能比整体用户数的类别少，导致产生空值
    # 为此需要进行空值去除，具体就是把第0列的空值去掉
    user_data = user_data.dropna(subset=[0])
    # codes是类别数值到索引值的映射，把原来很长的用户id映射成从0开始的位置整数
    user_data[0] = user_data[0].cat.codes
    user_data = user_data.sort_values(0)

    # 这里操作的含义就是缩减项目数，让第0列的index数值符合实际的item数量
    item_data[0] = item_data[0].astype('category')
    item_data[0] = item_data[0].cat.set_categories(train_data['item_id'].cat.categories)
    item_data = item_data.dropna(subset=[0])
    item_data[0] = item_data[0].cat.codes
    item_data = item_data.sort_values(0)

    # 处理用户以及项目的特征one-hot向量
    user_data[1] = user_data[1].astype('category')

    user_gender = user_data[1].cat.codes.values
    num_user_genders = len(user_data[1].cat.categories)

    item_data[1] = item_data[1].astype('category')

    item_genres = item_data[1].cat.codes.values
    num_item_genres = len(item_data[1].cat.categories)

    # 将上述特征赋予图中的结点
    graph.nodes['user'].data['gender'] = torch.LongTensor(user_gender)

    graph.nodes['item'].data['genres'] = torch.LongTensor(item_genres)

    # 本来直接用边类型就可以，但是会报错，只好用全称
    graph.edges[('item', 'watchedby', 'user')].data['rating'] = torch.LongTensor(train_ratings)
    graph.edges[('user', 'watched', 'item')].data['rating'] = torch.LongTensor(train_ratings)
    # -------------------------------------------------------------------------------------------------------------------
    # 创建用户、项目全连接图
    all_graph = dgl.heterograph({('user', 'watched', 'item'): (all_user_ids, all_item_ids)})
    # real_data为之后还原id做准备
    real_data = torch.tensor(list(zip(all_user_ids, all_item_ids)), dtype=torch.int)
    all_graph.edata['real_data'] = real_data
    # ---------------------------------------------从全连接图中去掉历史连接---------------------------------------------
    # 训练时要去掉用户和项目间的关联
    seeds = {'user': list(set(train_user_ids.tolist())),
             'item': list(set(train_item_ids.tolist()))}

    sampled_graph = all_graph.in_subgraph(seeds)

    _, _, edges_to_remove = sampled_graph.edge_ids(
        train_user_ids, train_item_ids, etype=('user', 'watched', 'item'), return_uv=True)
    # _, _, edges_to_remove_rev = graph.edge_ids(
    #     train_item_ids, train_user_ids, etype=('item', 'watchedby', 'user'), return_uv=True)

    # sampled_with_edges_removed = dgl.remove_edges(
    #     sampled_graph,
    #     {('user', 'watched', 'item'): edges_to_remove, ('item', 'watchedby', 'user'): edges_to_remove_rev}
    # )

    target_graph = dgl.remove_edges(sampled_graph, edges_to_remove, ('user', 'watched', 'item'))

    # target_graph = dgl.remove_edges(target_graph,edges_to_remove_rev, ('item', 'watchedby', 'user'))
    # ---------------------------------------------从全连接图中去掉历史连接---------------------------------------------
    # target_graph.nodes['user'].data['gender'] = torch.LongTensor(user_gender)
    # target_graph.nodes['item'].data['genres'] = torch.FloatTensor(item_genres)
    # -------------------------------------------------------------------------------------------------------------------
    # 设置数据集
    train_dataset = TensorDataset(train_user_ids, train_item_ids, train_ratings)

    # 定义训练过程
    NUM_LAYERS = layers
    BATCH_SIZE = batch_size
    NUM_EPOCHS = epochs

    HIDDEN_DIMS = hiddeen_dims
    sampler = MinibatchSampler(graph, NUM_LAYERS)

    # 准备加载数据
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=sampler.sample, shuffle=True)

    # 创建模型
    model = GCMCRating1(graph.number_of_nodes('user'), graph.number_of_nodes('item'), HIDDEN_DIMS, NUM_RATINGS,
                        NUM_LAYERS,
                        num_user_genders, num_item_genres)
    # 使用Adam优化器
    opt = torch.optim.Adam(model.parameters())

    epoch = 0
    # 开始训练
    for _ in range(NUM_EPOCHS):
        model.train()
        # 加个进度条，直观
        # 首先是训练
        with tqdm.tqdm(train_dataloader) as t:
            predictions = []
            # ratings = []
            # real_data = []
            for pair_graph, blocks in t:
                # real_data.append(pair_graph.edata['real_data'])
                user_emb, item_emb = model(blocks)
                prediction = model.compute_score(pair_graph, user_emb, item_emb)
                loss = ((prediction - pair_graph.edata['rating']) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                t.set_postfix({'loss': '%.4f' % loss.item()}, refresh=False)
                # ratings.append(pair_graph.edata['rating'])
                predictions.append(prediction)
            predictions = torch.cat(predictions, 0)
            # ratings = torch.cat(ratings, 0)
            # real_data = torch.cat(real_data, 0)
        model.eval()

        epoch += 1
        if epoch % 10 == 0:
            # -----------------------------------------------------------------------------------------------------------
            # ---------------------------------------graph转block---------------------------------------------
            # 创建子图块
            train_blocks = []
            block = dgl.to_block(graph)
            # 把评分复制过去
            # block.edges[('user', 'watched', 'item')].data['rating'] = \
            #     graph.edges[('user', 'watched', 'item')].data['rating']
            # block.edges[('item', 'watchedby', 'user')].data['rating'] = \
            #     graph.edges[('item', 'watchedby', 'user')].data['rating']

            train_blocks.insert(0, block)
            # ---------------------------------------graph转block---------------------------------------------

            # 将train_blocks输入模型
            user_emb, item_emb = model(train_blocks)

            # 基于target_graph得到预测评分
            prediction = model.compute_score(target_graph, user_emb, item_emb)

            # ---------------------------------------还原用户与项目的初始id---------------------------------------------
            real_data = target_graph.edata['real_data']

            real_data = pd.DataFrame(real_data.tolist())
            real_data.columns = ['user_id', 'item_id']

            real_user = real_data['user_id'].values.tolist()
            real_uid = [user_dict[k] for k in real_user]
            real_data['user_id'] = real_uid

            real_item = real_data['item_id'].values.tolist()
            real_uid = [item_dict[k] for k in real_item]
            real_data['item_id'] = real_uid
            # ---------------------------------------还原用户与项目的初始id---------------------------------------------
            # 将边两端的节点给到result
            result = real_data
            # 列表降维，去掉prediction中单个元素外的中括号
            predictions = np.ravel(prediction.tolist())
            # predictions = sum(prediction.tolist(), [])

            # 将边上预测得到的值给到result的'rating'
            result['rating'] = predictions
            # 按user_id分组排序
            result = result.groupby('user_id').apply(lambda x: x.sort_values(by="rating", ascending=False)).reset_index(
                drop=True)
            result.to_csv('file_saved/fsl-DGLresult.csv', index=None)
            # -----------------------------------------------------------------------------------------------------------
            print(result)

            result.to_csv('new_saved/dgl/fsl-DGLresult-epoch{}.csv'.format(epoch), index=None)

            # 以下在试图存储模型中的参数，可以注释掉
            # m1 = pd.DataFrame(model.W.weight.tolist())
            # m2 = pd.DataFrame(model.V.weight.tolist())
            # m1.to_csv('new_saved/dgl/fsl-GCMC-W-epoch{}.csv'.format(epoch), index=None)
            # m2.to_csv('new_saved/dgl/fsl-GCMC-V-epoch{}.csv'.format(epoch), index=None)
            #
            # for i in range(layers):
            #     l1 = pd.DataFrame(model.layers[i].heteroconv.mods['watchedby'].W_r.flatten(1).tolist())
            #     l2 = pd.DataFrame(model.layers[i].heteroconv.mods['watchedby'].W.weight.tolist())
            #     l1.to_csv('new_saved/dgl/fsl-GCMCConv-W_r-epoch{}-layer{}.csv'.format(epoch, i), index=None)
            #     l2.to_csv('new_saved/dgl/fsl-GCMCConv-W-epoch{}-layer{}.csv'.format(epoch, i), index=None)

        # 计算分类的覆盖情况，默认注释掉，不用跑
        # recommend(item_data_for_recommend, topK, FSLflag, classify_num=classify_num)


def dglMainMovielens(layers, batch_size, epochs, hiddeen_dims, topK):
    # 拿到数据
    train_data, test_data, user_data, item_data = get_ml_100k()

    # 把用户id和项目id的类别换成枚举类
    train_data = train_data.astype({'user_id': 'category', 'item_id': 'category'})
    test_data = test_data.astype({'user_id': 'category', 'item_id': 'category'})

    # 方便后续计算推荐结果
    item_data_for_recommend = item_data.copy()

    # 训练集和测试集的数据保持一致
    # test_data['user_id'].cat.set_categories(train_data['user_id'].cat.categories, inplace=True)
    # test_data['item_id'].cat.set_categories(train_data['item_id'].cat.categories, inplace=True)

    # 实现了绝对id到相对id的映射
    train_user_ids = torch.LongTensor(train_data['user_id'].cat.codes.values)
    train_item_ids = torch.LongTensor(train_data['item_id'].cat.codes.values)
    train_ratings = torch.LongTensor(train_data['rating'].values)

    # 全连接图的建立，准备user的id，item的id，item的id只留训练过的
    all_user_ids = list(set(train_user_ids.tolist()))
    all_item_ids = list(set(train_item_ids.tolist()))
    all_user_ids = [val for val in all_user_ids for i in range(len(all_item_ids))]
    all_item_ids = all_item_ids * len(set(all_user_ids))

    # 建立字典对应前后id
    user_dict = dict(zip(train_user_ids.tolist(), train_data['user_id'].values.tolist()))
    item_dict = dict(zip(train_item_ids.tolist(), train_data['item_id'].values.tolist()))

    # test_user_ids = torch.LongTensor(test_data['user_id'].cat.codes.values)
    # test_item_ids = torch.LongTensor(test_data['item_id'].cat.codes.values)
    # test_ratings = torch.LongTensor(test_data['rating'].values)

    # 创建异构图
    graph = dgl.heterograph({
        ('user', 'watched', 'item'): (train_user_ids, train_item_ids),
        ('item', 'watchedby', 'user'): (train_item_ids, train_user_ids)
    })

    # 令训练数据和用户、项目数据一致
    user_data[0] = user_data[0].astype('category')
    user_data[0] = user_data[0].cat.set_categories(train_data['user_id'].cat.categories)
    user_data = user_data.dropna(subset=[0])
    user_data[0] = user_data[0].cat.codes
    user_data = user_data.sort_values(0)

    item_data[0] = item_data[0].astype('category')
    item_data[0] = item_data[0].cat.set_categories(train_data['item_id'].cat.categories)
    item_data = item_data.dropna(subset=[0])
    item_data[0] = item_data[0].cat.codes
    item_data = item_data.sort_values(0)

    # 处理用户的年龄、性别、职业，以及项目的体裁one-hot向量
    user_data[2] = user_data[2].astype('category')
    user_data[3] = user_data[3].astype('category')
    # user_data[4] = user_data[4].astype('category')

    # user_age = user_data[1].values // 10
    # num_user_age_bins = user_age.max() + 1

    user_gender = user_data[2].cat.codes.values
    num_user_genders = len(user_data[2].cat.categories)

    # user_occupation = user_data[3].cat.codes.values
    # num_user_occupations = len(user_data[3].cat.categories)

    item_genres = item_data[range(5, 24)].values
    num_item_genres = item_genres.shape[1]

    # 将上述特征赋予图中的结点
    # graph.nodes['user'].data['age'] = torch.LongTensor(user_age)
    graph.nodes['user'].data['gender'] = torch.LongTensor(user_gender)
    # graph.nodes['user'].data['occupation'] = torch.LongTensor(user_occupation)

    graph.nodes['item'].data['genres'] = torch.FloatTensor(item_genres)

    # 本来直接用边类型就可以，但是会报错，只好用全称
    graph.edges[('item', 'watchedby', 'user')].data['rating'] = torch.LongTensor(train_ratings)
    graph.edges[('user', 'watched', 'item')].data['rating'] = torch.LongTensor(train_ratings)

    # target_graph.edges[('item', 'watchedby', 'user')].data['rating'] = torch.LongTensor(train_ratings)
    # target_graph.edges[('user', 'watched', 'item')].data['rating'] = torch.LongTensor(train_ratings)

    # 创建用户、项目全连接图
    all_graph = dgl.heterograph({('user', 'watched', 'item'): (all_user_ids, all_item_ids)})
    # real_data为之后还原id做准备
    real_data = torch.tensor(list(zip(all_user_ids, all_item_ids)), dtype=torch.int)
    all_graph.edata['real_data'] = real_data
    # ---------------------------------------------从全连接图中去掉历史连接------------------------------------------------
    # 训练时要去掉用户和项目间的关联
    seeds = {'user': list(set(train_user_ids.tolist())),
             'item': list(set(train_item_ids.tolist()))}

    sampled_graph = all_graph.in_subgraph(seeds)

    _, _, edges_to_remove = sampled_graph.edge_ids(
        train_user_ids, train_item_ids, etype=('user', 'watched', 'item'), return_uv=True)
    # _, _, edges_to_remove_rev = graph.edge_ids(
    #     train_item_ids, train_user_ids, etype=('item', 'watchedby', 'user'), return_uv=True)

    # sampled_with_edges_removed = dgl.remove_edges(
    #     sampled_graph,
    #     {('user', 'watched', 'item'): edges_to_remove, ('item', 'watchedby', 'user'): edges_to_remove_rev}
    # )

    target_graph = dgl.remove_edges(sampled_graph, edges_to_remove, ('user', 'watched', 'item'))

    # target_graph = dgl.remove_edges(target_graph,edges_to_remove_rev, ('item', 'watchedby', 'user'))
    # ---------------------------------------------从全连接图中去掉历史连接------------------------------------------------
    # target_graph.nodes['user'].data['gender'] = torch.LongTensor(user_gender)
    # target_graph.nodes['item'].data['genres'] = torch.FloatTensor(item_genres)

    # 设置数据集
    train_dataset = TensorDataset(train_user_ids, train_item_ids, train_ratings)

    # test_dataset = TensorDataset(test_user_ids, test_item_ids, test_ratings)

    # 定义训练过程
    NUM_LAYERS = layers
    BATCH_SIZE = batch_size
    NUM_EPOCHS = epochs

    HIDDEN_DIMS = hiddeen_dims
    sampler = MinibatchSampler(graph, NUM_LAYERS)

    # 准备加载数据
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=sampler.sample, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=sampler.sample, shuffle=False)

    # 创建模型
    model = GCMCRating2(graph.number_of_nodes('user'), graph.number_of_nodes('item'), HIDDEN_DIMS, 5, NUM_LAYERS,
                        num_user_genders, num_item_genres)

    # 使用Adam优化器
    opt = torch.optim.Adam(model.parameters())

    # 开始训练
    epoch = 0
    for _ in range(NUM_EPOCHS):
        model.train()
        # 加个进度条，直观
        # 首先是训练
        with tqdm.tqdm(train_dataloader) as t:
            # with torch.no_grad():
            predictions = []
            ratings = []
            # real_data = []
            for pair_graph, blocks in t:
                # real_data.append(pair_graph.edata['real_data'])
                user_emb, item_emb = model(blocks)
                prediction = model.compute_score(pair_graph, user_emb, item_emb)
                loss = ((prediction - pair_graph.edata['rating']) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                t.set_postfix({'loss': '%.4f' % loss.item()}, refresh=False)
                ratings.append(pair_graph.edata['rating'])
                predictions.append(prediction)
            # predictions = torch.cat(predictions, 0)
            # ratings = torch.cat(ratings, 0)
            # real_data = torch.cat(real_data, 0)
        model.eval()

        epoch += 1
        if epoch + 1 == NUM_EPOCHS:
            # if epoch % 10 == 0:
            # -----------------------------------------------------------------------------------------------------------
            # ---------------------------------------graph转block---------------------------------------------
            # 创建子图块
            train_blocks = []
            block = dgl.to_block(graph)
            # 把评分复制过去
            # block.edges[('user', 'watched', 'item')].data['rating'] = \
            #     graph.edges[('user', 'watched', 'item')].data['rating']
            # block.edges[('item', 'watchedby', 'user')].data['rating'] = \
            #     graph.edges[('item', 'watchedby', 'user')].data['rating']

            train_blocks.insert(0, block)
            # ---------------------------------------graph转block---------------------------------------------

            # 将train_blocks输入模型
            user_emb, item_emb = model(train_blocks)

            # 基于target_graph得到预测评分
            prediction = model.compute_score(target_graph, user_emb, item_emb)

            # ---------------------------------------还原用户与项目的初始id---------------------------------------------
            real_data = target_graph.edata['real_data']

            real_data = pd.DataFrame(real_data.tolist())
            real_data.columns = ['user_id', 'item_id']

            real_user = real_data['user_id'].values.tolist()
            real_uid = [user_dict[k] for k in real_user]
            real_data['user_id'] = real_uid

            real_item = real_data['item_id'].values.tolist()
            real_uid = [item_dict[k] for k in real_item]
            real_data['item_id'] = real_uid
            # ---------------------------------------还原用户与项目的初始id---------------------------------------------
            # 将边两端的节点给到result
            result = real_data
            # 列表降维，去掉prediction中单个元素外的中括号
            predictions = np.ravel(prediction.tolist())
            # predictions = sum(prediction.tolist(), [])

            # 将边上预测得到的值给到result的'rating'
            result['rating'] = predictions
            # 按user_id分组排序
            result = result.groupby('user_id').apply(lambda x: x.sort_values(by="rating", ascending=False)).reset_index(
                drop=True)
            # result.to_csv('file_saved/ml-DGLresult.csv', index=None)
            # -----------------------------------------------------------------------------------------------------------
            # print(result)

            result.to_csv('new_saved/dgl/ml-DGLresult-epoch{}.csv'.format(epoch), index=None)

            # 读取用户与项目的向量并存储
            # m1 = pd.DataFrame(model.W.weight.tolist())
            # m2 = pd.DataFrame(model.V.weight.tolist())
            # m1.to_csv('new_saved/dgl/ml-GCMC-W-epoch{}.csv'.format(epoch), index=None,header=None)
            # m2.to_csv('new_saved/dgl/ml-GCMC-V-epoch{}.csv'.format(epoch), index=None,header=None)

            # 读取模型的神经网络参数并存储
            # for i in range(layers):
            #     l1 = pd.DataFrame(model.layers[i].heteroconv.mods['watchedby'].W_r.flatten(1).tolist())
            #     l2 = pd.DataFrame(model.layers[i].heteroconv.mods['watchedby'].W.weight.tolist())
            #     l1.to_csv('new_saved/dgl/ml-GCMCConv-W_r-epoch{}-layer{}.csv'.format(epoch,i), index=None,header=None)
            #     l2.to_csv('new_saved/dgl/ml-GCMCConv-W-epoch{}-layer{}.csv'.format(epoch, i), index=None,header=None)

        # recommend(item_data_for_recommend, topK , FSLflag)


def run():
    if FSLflag == False:
        dglMainMovielens(layers=1, batch_size=500, epochs=50, hiddeen_dims=8, topK=10)
    else:
        # dglMainFSL(layers=1, batch_size=500, epochs=1, hiddeen_dims=8, topK=10)
        # 报错KeyError:'user' 或者 'item'，要修改batch_size
        # 前5000的数据量太小，需要降低采样数量，batch_size就要设置得小一些，例如设为5，数据多了再改
        dglMainFSL(layers=1, batch_size=5, epochs=50, hiddeen_dims=8, topK=10)
