import dgl
import torch


# 实现采样和子图的提取
class MinibatchSampler(object):
    def __init__(self, graph, num_layers):
        self.graph = graph
        self.num_layers = num_layers

    def sample(self, batch):
        users, items, ratings = zip(*batch)
        users = torch.stack(users)
        items = torch.stack(items)
        ratings = torch.stack(ratings)
        # 1 创建二部图
        pair_graph = dgl.heterograph(
            {('user', 'watched', 'item'): (users, items)},
            num_nodes_dict={'user': self.graph.num_nodes('user'),
                            'item': self.graph.num_nodes('item')}
        )

        u = users.tolist()
        i = items.tolist()
        real_data = torch.tensor(list(zip(u,i)),dtype = torch.int)
        pair_graph.edata['real_data'] = real_data


        # 2 压缩二部图
        pair_graph = dgl.compact_graphs(pair_graph)
        pair_graph.edata['rating'] = ratings

        # 3 创建数据块
        seeds = {'user': pair_graph.nodes['user'].data[dgl.NID],
                 'item': pair_graph.nodes['item'].data[dgl.NID]}
        blocks = self.construct_blocks(seeds, (users, items))

        # 把节点特征也复制过来
        # 注意这里只需要处理源端结点
        for feature_name in self.graph.nodes['user'].data.keys():
            blocks[0].srcnodes['user'].data[feature_name] = \
                self.graph.nodes['user'].data[feature_name][blocks[0].srcnodes['user'].data[dgl.NID]]
        for feature_name in self.graph.nodes['item'].data.keys():
            blocks[0].srcnodes['item'].data[feature_name] = \
                self.graph.nodes['item'].data[feature_name][blocks[0].srcnodes['item'].data[dgl.NID]]

        return pair_graph, blocks

    def construct_blocks(self, seeds, user_item_pairs_to_remove):
        blocks = []
        users, items = user_item_pairs_to_remove
        # 采样就是根据卷积层数选取对应数量的邻居结点
        # 涉及到双向图的处理
        for i in range(self.num_layers):
            sampled_graph = dgl.in_subgraph(self.graph, seeds)
            sampled_eids = sampled_graph.edges[('user', 'watched', 'item')].data[dgl.EID]
            sampled_eids_rev = sampled_graph.edges[('item', 'watchedby', 'user')].data[dgl.EID]

            # 训练时要去掉用户和项目间的关联
            _, _, edges_to_remove = sampled_graph.edge_ids(
                users, items, etype=('user', 'watched', 'item'), return_uv=True)
            _, _, edges_to_remove_rev = sampled_graph.edge_ids(
                items, users, etype=('item', 'watchedby', 'user'), return_uv=True)

            # sampled_with_edges_removed = dgl.remove_edges(
            #     sampled_graph,
            #     {('user', 'watched', 'item'): edges_to_remove, ('item', 'watchedby', 'user'): edges_to_remove_rev}
            # )

            sampled_with_edges_removed = dgl.remove_edges(sampled_graph,
                                                          edges_to_remove, ('user', 'watched', 'item'))
            sampled_with_edges_removed = dgl.remove_edges(sampled_with_edges_removed,
                                                          edges_to_remove_rev, ('item', 'watchedby', 'user'))

            sampled_eids = sampled_eids[
                sampled_with_edges_removed.edges[('user', 'watched', 'item')].data[dgl.EID]]
            sampled_eids_rev = sampled_eids_rev[
                sampled_with_edges_removed.edges[('item', 'watchedby', 'user')].data[dgl.EID]]

            # 创建子图块
            block = dgl.to_block(sampled_with_edges_removed, seeds)
            blocks.insert(0, block)
            seeds = {'user': block.srcnodes['user'].data[dgl.NID],
                     'item': block.srcnodes['item'].data[dgl.NID]
                     }

            # 把评分复制过去
            block.edges[('user', 'watched', 'item')].data['rating'] = \
                self.graph.edges[('user', 'watched', 'item')].data['rating'][sampled_eids]
            block.edges[('item', 'watchedby', 'user')].data['rating'] = \
                self.graph.edges[('item', 'watchedby', 'user')].data['rating'][sampled_eids_rev]

        return blocks
