import dgl
import torch
from DGL.mlload import get_ml_100k,get_fshl,get_bigraph
from torch.utils.data import TensorDataset, DataLoader
from DGL.minibatch import MinibatchSampler
from DGL.gcmc import rmse, GCMCRating1, GCMCRating2
import tqdm
import pandas as pd


FSLflag = False

def makeClassifyDict(item_data,FSLflag):
    print("make classify dict")
    # 示例{“课程id”：[类别1、类别2、类别3]}
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

def recommend(item_data,topK,FSLflag,classify_num=0):
    if FSLflag == False:
        result = pd.read_csv('file_saved/ml-DGLresult.csv')
    else:
        result = pd.read_csv('file_saved/fsl-DGLresult.csv')
    result_topK = result.groupby(['user_id']).head(topK).reset_index(drop=True)

    # 计算列表覆盖率
    recommend_length = len(result_topK['item_id'].value_counts())
    item_length = len(item_data)
    cov = recommend_length/item_length
    print("the rate of coverage: ")
    print(cov)

    # 计算物品类别覆盖率
    # 记录推荐类别的字典
    classify_num_dict = {}
    classify = makeClassifyDict(item_data,FSLflag)
    item_id = result_topK['item_id'].value_counts().values.tolist()
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
    if FSLflag == False:
        classify_cov = (len(classify_num_dict) * 1.0) / 19.0
    else:
        classify_num = classify_num
        classify_cov = (len(classify_num_dict) * 1.0) / classify_num
    print("the rate of classify coverage: ")
    print(classify_cov)

def dglMainFSL(layers,batch_size,epochs,hiddeen_dims,topK):
    # 从初始输入里拿到数据
    train_data, user_data, item_data,classify_data = get_bigraph()
    item_data_for_recommend = item_data
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

    # 建立字典对应前后id
    user_dict = dict(zip(train_user_ids.tolist(),train_data['user_id'].values.tolist()))
    item_dict = dict(zip(train_item_ids.tolist(),train_data['item_id'].values.tolist()))

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
    model = GCMCRating1(graph.number_of_nodes('user'), graph.number_of_nodes('item'), HIDDEN_DIMS, 5, NUM_LAYERS,
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
            real_data = []
            for pair_graph, blocks in t:
                real_data.append(pair_graph.edata['real_data'])
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
            real_data = torch.cat(real_data, 0)
        model.eval()

        #取出真实id，并用字典映射回去，生成推荐列表
        real_data = pd.DataFrame(real_data.tolist())
        real_data.columns = ['user_id', 'item_id']

        real_user = real_data['user_id'].values.tolist()
        real_uid = [user_dict[k] for k in real_user]
        real_data['user_id'] = real_uid

        real_item = real_data['item_id'].values.tolist()
        real_uid = [item_dict[k] for k in real_item]
        real_data['item_id'] = real_uid
        result = real_data
        # print(predictions)
        predictions = sum(predictions.tolist(), [])
        result['rating'] = predictions
        result = result.groupby('user_id').apply(lambda x: x.sort_values(by="rating", ascending=False)).reset_index(
            drop=True)
        result.to_csv('file_saved/fsl-DGLresult.csv', index=None)
        print(result)

        recommend(item_data_for_recommend, topK , FSLflag , classify_num=classify_num)


def dglMainMovielens(layers,batch_size,epochs,hiddeen_dims,topK):
    # 拿到数据
    train_data, test_data, user_data, item_data = get_ml_100k()
    item_data_for_recommend = item_data
    # 把用户id和项目id的类别换成枚举类
    train_data = train_data.astype({'user_id': 'category', 'item_id': 'category'})
    test_data = test_data.astype({'user_id': 'category', 'item_id': 'category'})

    # 训练集和测试集的数据保持一致
    # test_data['user_id'].cat.set_categories(train_data['user_id'].cat.categories, inplace=True)
    # test_data['item_id'].cat.set_categories(train_data['item_id'].cat.categories, inplace=True)

    # 实现了绝对id到相对id的映射
    train_user_ids = torch.LongTensor(train_data['user_id'].cat.codes.values)
    train_item_ids = torch.LongTensor(train_data['item_id'].cat.codes.values)
    train_ratings = torch.LongTensor(train_data['rating'].values)

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
    for _ in range(NUM_EPOCHS):
        model.train()
        # 加个进度条，直观
        # 首先是训练
        with tqdm.tqdm(train_dataloader) as t:
            # with torch.no_grad():
            predictions = []
            ratings = []
            real_data = []
            for pair_graph, blocks in t:
                real_data.append(pair_graph.edata['real_data'])
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
            real_data = torch.cat(real_data, 0)
        model.eval()
        real_data = pd.DataFrame(real_data.tolist())
        real_data.columns = ['user_id', 'item_id']

        real_user = real_data['user_id'].values.tolist()
        real_uid = [user_dict[k] for k in real_user]
        real_data['user_id'] = real_uid

        real_item = real_data['item_id'].values.tolist()
        real_uid = [item_dict[k] for k in real_item]
        real_data['item_id'] = real_uid
        result = real_data
        predictions = sum(predictions.tolist(), [])
        result['rating'] = predictions
        result = result.groupby('user_id').apply(lambda x: x.sort_values(by="rating", ascending=False)).reset_index(
            drop=True)
        result.to_csv('file_saved/ml-DGLresult.csv', index=None)
        print(result)
        recommend(item_data_for_recommend, topK , FSLflag)

def run():
    if FSLflag == False:
        dglMainMovielens(layers=1,batch_size=500,epochs=1,hiddeen_dims=8,topK=10)
    else:
        dglMainFSL(layers=1,batch_size=500,epochs=1,hiddeen_dims=8,topK=10)

