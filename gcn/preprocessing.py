# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import cPickle as pkl
import os
import h5py
import pandas as pd
import csv

from data_utils import load_data, map_data, download_dataset
from feature import makeFeature


def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    if len(degree) == 610:
        pass
    else:
        degree[degree == 0] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm


def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """
    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    u_features = sp.hstack([u_features, zero_csr_u], format='csr')
    v_features = sp.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features


def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I

    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm


def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def create_trainvaltest_split(dataset, seed=1234, testing=False, datasplit_path=None, datasplit_from_file=False,
                              verbose=True):
    """
    Splits data set into train/val/test sets from full bipartite adjacency matrix. Shuffling of dataset is done in
    load_data function.
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix.
    """

    if datasplit_from_file and os.path.isfile(datasplit_path):
        print('Reading dataset splits from file...')
        with open(datasplit_path) as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = pkl.load(f)

        if verbose:
            print('Number of users = %d' % num_users)
            print('Number of items = %d' % num_items)
            print('Number of links = %d' % ratings.shape[0])
            print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    else:
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load_data(dataset, seed=seed,
                                                                                            verbose=verbose)

        with open(datasplit_path, 'w') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features], f)

    neutral_rating = -1

    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    labels = labels.reshape([-1])

    # number of test and validation edges
    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    if dataset == 'fshl':
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))


    num_train = ratings.shape[0] - num_val - num_test

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])

    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    train_idx = idx_nonzero[0:num_train]
    val_idx = idx_nonzero[num_train:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    train_pairs_idx = pairs_nonzero[0:num_train]
    val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    class_values = np.sort(np.unique(ratings))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values


def load_data_monti(dataset, testing=False):
    """
    Loads data from Monti et al. paper.
    """

    path_dataset = 'data/' + dataset.replace('_', '-') + '/training_test_dataset.mat'

    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')

    num_users = M.shape[0]
    num_items = M.shape[1]

    if dataset == 'flixster':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        Wcol = load_matlab_file(path_dataset, 'W_movies')
        u_features = Wrow
        v_features = Wcol
        # print(num_items, v_features.shape)
        # v_features = np.eye(num_items)

    elif dataset == 'douban':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        u_features = Wrow
        v_features = np.eye(num_items)
    elif dataset == 'yahoo_music':
        Wcol = load_matlab_file(path_dataset, 'W_tracks')
        u_features = np.eye(num_users)
        v_features = Wcol

    u_nodes_ratings = np.where(M)[0]
    v_nodes_ratings = np.where(M)[1]
    ratings = M[np.where(M)]

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    print('number of users = ', len(set(u_nodes)))
    print('number of item = ', len(set(v_nodes)))

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges

    num_train = np.where(Otraining)[0].shape[0]
    num_test = np.where(Otest)[0].shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero_train = np.array([[u, v] for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1])])
    idx_nonzero_train = np.array([u * num_items + v for u, v in pairs_nonzero_train])

    pairs_nonzero_test = np.array([[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
    idx_nonzero_test = np.array([u * num_items + v for u, v in pairs_nonzero_test])

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = range(len(idx_nonzero_train))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    class_values = np.sort(np.unique(ratings))

    if u_features is not None:
        u_features = sp.csr_matrix(u_features)
        print("User features shape: " + str(u_features.shape))

    if v_features is not None:
        v_features = sp.csr_matrix(v_features)
        print("Item features shape: " + str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values


def load_official_trainvaltest_split(dataset, testing=False):
    """
    Loads official train/test split and uses 10% of training samples for validaiton
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
    """
    sep = ','

    # Check if files exist and download otherwise
    files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    fname = dataset
    data_dir = 'data/' + fname

    ### here we make this download operation unavailable, use local files instead
    # download_dataset(fname, files, data_dir)

    dtypes = {
        'u_nodes': np.int64, 'v_nodes': np.int64,
        'ratings': np.float32}

    # filename_train = 'mat.csv'
    # filename_test = 'mat.csv'

    # 数据输入GCN前进行一次转换,手动构造dataframe
    u_nodes, v_nodes, ratings = [], [], []
    i = 0
    with open('mat.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            u_nodes.append(int(row[0]))
            v_nodes.append(int(row[1]))
            ratings.append(int(row[2]))

    uSuperDict = {r: i for i, r in enumerate(list(set(u_nodes)))}
    vSuperDict = {r: i for i, r in enumerate(list(set(v_nodes)))}
    print(uSuperDict)
    #保存字典
    u_listKey = []
    u_listValue = []
    for key in uSuperDict:
        u_listKey.append(uSuperDict[key])
        u_listValue.append(key)

    u_to_dictr = zip(u_listKey, u_listValue)
    u_dictr = dict((u_listKey, u_listValue) for u_listKey, u_listValue in u_to_dictr)

    v_listKey = []
    v_listValue = []
    for key in vSuperDict:
        v_listKey.append(vSuperDict[key])
        v_listValue.append(key)
    v_to_dictr = zip(v_listKey, v_listValue)
    v_dictr = dict((v_listKey, v_listValue) for v_listKey, v_listValue in v_to_dictr)

    np.save('u_dictr.npy', u_dictr)
    np.save('v_dictr.npy', v_dictr)

    new_u_nodes, new_v_nodes = [], []
    for uid in u_nodes:
        new_u_nodes.append(uSuperDict[uid])
    for vid in v_nodes:
        new_v_nodes.append(vSuperDict[vid])
    u_nodes, v_nodes = new_u_nodes, new_v_nodes

    data_dict = {
        'u_nodes': np.int64(u_nodes),
        'v_nodes': np.int64(v_nodes),
        'ratings': np.float32(ratings)
    }
    data_train = pd.DataFrame(data=data_dict)
    data_test = pd.DataFrame(data=data_dict)

    # data_train = pd.read_csv(
    #     filename_train, sep=sep, header=None,
    #     names=['u_nodes', 'v_nodes', 'ratings'], dtype=dtypes)
    #
    # data_test = pd.read_csv(
    #     filename_test, sep=sep, header=None,
    #     names=['u_nodes', 'v_nodes', 'ratings'], dtype=dtypes)

    '''
    sep = '/t'
    # Check if files exist and download otherwise
    # files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    # fname = dataset
    # data_dir = 'data/' + fname
    # here we make this download operation unavailable, use local files instead
    # download_dataset(fname, files, data_dir)
    dtypes = {
        'u_nodes': np.int64, 'v_nodes': np.int32,
        'ratings': np.float32}
    filename_train = 'u1.base'
    filename_test = 'u1.test'
    data_train = pd.read_csv(
        filepath_or_buffer=filename_train, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings'], dtype=dtypes)
    data_test = pd.read_csv(
        filename_test, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings'], dtype=dtypes,
        engine='python')
    '''

    data_array_train = data_train.as_matrix().tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.as_matrix().tolist()
    data_array_test = np.array(data_array_test)

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 1].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 2].astype(dtypes['v_nodes'])
    ratings = data_array[:, 0].astype(dtypes['ratings'])

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
    print("num_users = {}".format(num_users))
    print("num_item = {}".format(num_items))

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}
    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)

    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert (labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code

    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    for i in range(len(ratings)):
        assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    idx_nonzero_train = idx_nonzero[0:num_train + num_val]
    idx_nonzero_test = idx_nonzero[num_train + num_val:]

    pairs_nonzero_train = pairs_nonzero[0:num_train + num_val]
    pairs_nonzero_test = pairs_nonzero[num_train + num_val:]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = range(len(idx_nonzero_train))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert (len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    class_values = np.sort(np.unique(ratings))

    if dataset == 'fshl':
        '''
        # movie features (genres)
        sep = r'|'
        movie_file = 'data/' + dataset.replace('_', '-') + '/u.item'
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')
        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]
        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec
        # user features
        sep = r'|'
        users_file = 'data/' + dataset.replace('_', '-') + '/u.user'
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')
        occupation = set(users_df['occupation'].values.tolist())
        age = users_df['age'].values
        age_max = age.max()
        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}
        num_feats = 2 + len(occupation_dict)
        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age'] / np.float(age_max)
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.
        '''
        u_features, v_features = makeFeature()
    else:
        raise ValueError('Invalid dataset option %s' % dataset)

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    print("User features shape: " + str(u_features.shape))
    print("Item features shape: " + str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, \
        u_train_idx, v_train_idx, val_labels, u_val_idx, v_val_idx, \
        test_labels, u_test_idx, v_test_idx, class_values, uSuperDict, vSuperDict


def new_train_split():
    # 此设置指定numpy在打印时输出全部元素
    np.set_printoptions(threshold=np.inf)

    # 数据输入GCN前进行一次转换,手动构造dataframe
    u_nodes, v_nodes, ratings = [], [], []
    i = 0
    # 注意这里要换用二部图的输出
    #with open('C:/Users/Administrator/Desktop/HybridRecommendGCN/gcn/toGcn.csv', 'r') as f:
    with open('gcn/toGcn.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            u_nodes.append(int(row[0]))
            v_nodes.append(int(row[1]))
            ratings.append(int(row[2]))

    # 构造ID映射字典，从长ID映射为0开始的数字
    uSuperDict = {r: i for i, r in enumerate(list(set(u_nodes)))}
    vSuperDict = {r: i for i, r in enumerate(list(set(v_nodes)))}

    # 构建用户反向字典，从数字映射回ID
    u_listKey = []
    u_listValue = []
    for key in uSuperDict:
        u_listKey.append(uSuperDict[key])
        u_listValue.append(key)

    u_to_dictr = zip(u_listKey, u_listValue)
    u_dictr = dict((u_listKey, u_listValue) for u_listKey, u_listValue in u_to_dictr)

    # 构造课程反向字典
    v_listKey = []
    v_listValue = []
    for key in vSuperDict:
        v_listKey.append(vSuperDict[key])
        v_listValue.append(key)
    v_to_dictr = zip(v_listKey, v_listValue)
    v_dictr = dict((v_listKey, v_listValue) for v_listKey, v_listValue in v_to_dictr)

    # 保存反向字典
    np.save('u_dictr.npy', u_dictr)
    np.save('v_dictr.npy', v_dictr)

    # 抽取出映射过的ID，作为系统输入
    new_u_nodes, new_v_nodes = [], []
    for uid in u_nodes:
        new_u_nodes.append(uSuperDict[uid])
    for vid in v_nodes:
        new_v_nodes.append(vSuperDict[vid])
    u_nodes, v_nodes = new_u_nodes, new_v_nodes

    data_dict = {
        'u_nodes': np.int64(u_nodes),
        'v_nodes': np.int64(v_nodes),
        'ratings': np.float32(ratings)
    }
    # 根据转换过的ID重新构建评分表
    data_array = pd.DataFrame(data=data_dict)

    # 转换为三元组的二维数组，每个元组为[评分，UID，VID]
    data_array = data_array.as_matrix().tolist()
    data_array = np.array(data_array)
    # print(data_array)

    # 设定数据类型字典
    dtypes = {
        'u_nodes': np.int64, 'v_nodes': np.int64,
        'ratings': np.float32}

    # 分离出用户ID、内容ID、评分3个向量
    u_nodes_ratings = data_array[:, 1].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 2].astype(dtypes['v_nodes'])
    ratings = data_array[:, 0].astype(dtypes['ratings'])
    # print(u_nodes_ratings, v_nodes_ratings, ratings)

    # 计算用户数量
    # 这里的字典和上边的重复了，没啥用，拟删除
    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    # print(u_nodes_ratings, u_dict)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
    print("num_users = {}".format(num_users))
    print("num_item = {}".format(num_items))

    # 转换数据类型
    u_nodes_ratings = u_nodes_ratings.astype(np.int64)
    v_nodes_ratings = v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    # 将转换后的ID作为输入
    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    # 假设每个评分等级至少有一条用户-课程交互数据
    # 去重整理出评分等级
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}
    # label数据初始化，用户数*课程数的矩阵，初始值为neutral_rating
    # label会作为后续训练集、测试集切分的数据来源
    neutral_rating = -1
    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    # 根据ratings赋值，构造出评分矩阵
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    # 验证数据是否相等
    for i in range(len(u_nodes)):
        assert (labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    # 化为向量
    labels = labels.reshape([-1])
    print(labels.shape)

    # 这里我们分出训练集和验证集，训练集是整体数据的1/5
    num_train = data_array.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    # 创建用户-课程对
    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    # 用户-课程对从矩阵压缩成向量后的绝对位置计算，也就是labels中的索引位置
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    # 验证转换是否正确
    for i in range(len(ratings)):
        assert (labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    # 保留训练集的索引和数据对
    idx_nonzero_train = idx_nonzero[0:num_train]
    # print(idx_nonzero_train.shape)
    pairs_nonzero_train = pairs_nonzero[0:num_train]

    # 将训练集打散，也就是用户-课程对打散，同时评分在labels里的位置也打散
    rand_idx = range(len(idx_nonzero_train))
    # print(rand_idx)
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    # print(idx_nonzero_train)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    # print(idx_nonzero_train)
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    # 把打散的训练集和测试集合成完整的集合
    # 目前的处理下测试集为空，实际上全是打散的训练集
    idx_nonzero = idx_nonzero_train
    pairs_nonzero = pairs_nonzero_train

    # 取出label中测试集元素对应的位置，以及用户-课程对
    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train+num_val]

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train+num_val]

    # 通过转置，把数据集中的用户和课程分离在向量中
    u_train_idx, v_train_idx = train_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()

    # 对存储评分的label向量进行相同的切分
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    # print(train_labels)

    # 建立评分向量，并将其变形为矩阵
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.

    # 进行矩阵压缩，得到矩阵三元组，即（i, j, Rij）
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    # print(rating_mx_train)

    # 去重取出评分等级
    class_values = np.sort(np.unique(ratings))
    print(class_values)

    # 根据特征建立用户与课程的描述向量,并稀疏化
    u_features, v_features = makeFeature()
    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    print("User features shape: " + str(u_features.shape))
    print("Item features shape: " + str(v_features.shape))

    # 最后返回全部数据
    return u_features, v_features, rating_mx_train, train_labels, \
        u_train_idx, v_train_idx, val_labels, u_val_idx, v_val_idx, \
        class_values, uSuperDict, vSuperDict


def get_original_labels():
    dtypes = {
        'u_nodes': np.int64, 'v_nodes': np.int64,
        'ratings': np.float32}

    # 数据输入GCN前进行一次转换,手动构造dataframe
    u_nodes, v_nodes, ratings = [], [], []
    i = 0
    # 注意这里要使用二部图的输入作为初始数据
    with open('bg_input.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            u_nodes.append(int(row[0]))
            v_nodes.append(int(row[1]))
            ratings.append(int(row[2]))

    # 构造ID映射字典，从长ID映射为0开始的数字
    uOriginalDict = {r: i for i, r in enumerate(list(set(u_nodes)))}
    vOriginalDict = {r: i for i, r in enumerate(list(set(v_nodes)))}

    # 构建用户反向字典，从数字映射回ID
    u_listKey = []
    u_listValue = []
    for key in uOriginalDict:
        u_listKey.append(uOriginalDict[key])
        u_listValue.append(key)

    u_to_dictr = zip(u_listKey, u_listValue)
    u_dictr = dict((u_listKey, u_listValue) for u_listKey, u_listValue in u_to_dictr)

    # 构造课程反向字典
    v_listKey = []
    v_listValue = []
    for key in vOriginalDict:
        v_listKey.append(vOriginalDict[key])
        v_listValue.append(key)
    v_to_dictr = zip(v_listKey, v_listValue)
    v_dictr = dict((v_listKey, v_listValue) for v_listKey, v_listValue in v_to_dictr)

    # 保存反向字典
    np.save('original_u_dict.npy', u_dictr)
    np.save('original_v_dict.npy', v_dictr)

    # 抽取出映射过的ID
    new_u_nodes, new_v_nodes = [], []
    for uid in u_nodes:
        new_u_nodes.append(uOriginalDict[uid])
    for vid in v_nodes:
        new_v_nodes.append(vOriginalDict[vid])
    u_nodes, v_nodes = new_u_nodes, new_v_nodes

    data_dict = {
        'u_nodes': np.int64(u_nodes),
        'v_nodes': np.int64(v_nodes),
        'ratings': np.float32(ratings)
    }
    # 根据转换过的ID重新构建评分表
    data_array = pd.DataFrame(data=data_dict)

    # 转换为三元组的二维数组，每个元组为[评分，UID，VID]
    data_array = data_array.as_matrix().tolist()
    data_array = np.array(data_array)
    # print(data_array)

    # 分离出用户ID、内容ID、评分3个向量
    u_nodes_ratings = data_array[:, 1].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 2].astype(dtypes['v_nodes'])
    ratings = data_array[:, 0].astype(dtypes['ratings'])
    # print(u_nodes_ratings, v_nodes_ratings, ratings)

    # 计算用户数量
    num_users = len(list(set(u_nodes_ratings)))
    num_items = len(list(set(v_nodes_ratings)))
    print("num_users = {}".format(num_users))
    print("num_item = {}".format(num_items))

    # 转换数据类型
    u_nodes_ratings = u_nodes_ratings.astype(np.int64)
    v_nodes_ratings = v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    # 将转换后的ID作为输入
    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    # 假设每个评分等级至少有一条用户-课程交互数据
    # 去重整理出评分等级
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}
    # label数据初始化，用户数*课程数的矩阵，初始值为neutral_rating
    # label会作为后续训练集、测试集切分的数据来源
    neutral_rating = -1
    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    # 根据ratings赋值，构造出评分矩阵
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    # 验证数据是否相等
    for i in range(len(u_nodes)):
        assert (labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    # 化为向量
    labels = labels.reshape([-1])
    # print(labels)

    # 这里我们分出训练集和验证集，训练集是整体数据的1/5
    num_train = data_array.shape[0]

    # 创建用户-课程对
    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    # 用户-课程对从矩阵压缩成向量后的绝对位置计算，也就是labels中的索引位置
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    # 验证转换是否正确
    for i in range(len(ratings)):
        assert (labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    # 保留训练集的索引和数据对
    idx_nonzero_train = idx_nonzero[0:num_train]
    # print(idx_nonzero_train.shape)
    pairs_nonzero_train = pairs_nonzero[0:num_train]

    # 将训练集打散，也就是用户-课程对打散，同时评分在labels里的位置也打散
    rand_idx = range(len(idx_nonzero_train))
    # print(rand_idx)
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    # print(idx_nonzero_train)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    # print(idx_nonzero_train)
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    # 把打散的训练集和测试集合成完整的集合
    # 目前的处理下测试集为空，实际上全是打散的训练集
    idx_nonzero = idx_nonzero_train
    pairs_nonzero = pairs_nonzero_train

    # 取出label中测试集元素对应的位置，以及用户-课程对
    train_idx = idx_nonzero[0:num_train]
    train_pairs_idx = pairs_nonzero[0:num_train]

    # 通过转置，把数据集中的用户和课程分离在向量中
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # 对存储评分的label向量进行相同的切分
    train_labels = labels[train_idx]

    # 最后返回二部图输入的训练集
    return train_labels, u_train_idx, v_train_idx,\
        uOriginalDict, vOriginalDict
