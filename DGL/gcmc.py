from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch


class GCMCConv(nn.Module):
    def __init__(self, hidden_dims, num_ratings):
        super().__init__()

        self.W_r = nn.Parameter(torch.randn(num_ratings + 1, hidden_dims, hidden_dims))
        self.W = nn.Linear(hidden_dims * 2, hidden_dims)

    def compute_message(self, W, edges):
        W_r = W[edges.data['rating']]
        h = edges.src['h']
        m = (W_r @ h.unsqueeze(-1)).squeeze(2)
        return m

    def forward(self, graph, node_features):
        with graph.local_scope():
            src_features, dst_features = node_features
            graph.srcdata['h'] = src_features
            graph.dstdata['h'] = dst_features

            graph.apply_edges(lambda edges: {'m': self.compute_message(self.W_r, edges)})

            graph.update_all(fn.copy_e('m', 'm'), fn.mean('m', 'h_neigh'))

            result = F.relu(self.W(torch.cat([graph.dstdata['h'], graph.dstdata['h_neigh']], 1)))

            return result


class GCMCLayer(nn.Module):
    def __init__(self, hidden_dims, num_ratings):
        super().__init__()

        self.heteroconv = dglnn.HeteroGraphConv(
            {'watchedby': GCMCConv(hidden_dims, num_ratings),
             'watched': GCMCConv(hidden_dims, num_ratings),
             },
            aggregate='sum'
        )

    def forward(self, block, input_user_features, input_item_features):
        with block.local_scope():
            h_user = input_user_features
            h_item = input_item_features

            src_features = {'user': h_user, 'item': h_item}
            dst_features = {'user': h_user[:block.number_of_dst_nodes('user')],
                            'item': h_item[:block.number_of_dst_nodes('item')]}

            result = self.heteroconv(block, (src_features, dst_features))
            return result['user'], result['item']

#fsl
class GCMCRating1(nn.Module):
    # def __init__(self, num_users, num_items, hidden_dims, num_ratings, num_layers,
    #              num_user_age_bins, num_user_genders, num_user_occupations, num_item_genres):
    def __init__(self, num_users, num_items, hidden_dims, num_ratings, num_layers,
                 num_user_genders, num_item_genres):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, hidden_dims)
        self.item_embeddings = nn.Embedding(num_items, hidden_dims)

        # self.U_age = nn.Embedding(num_user_age_bins, hidden_dims)
        self.U_gender = nn.Embedding(num_user_genders, hidden_dims)
        self.I_genres = nn.Embedding(num_item_genres, hidden_dims)
        self.layers = nn.ModuleList([
            GCMCLayer(hidden_dims, num_ratings) for _ in range(num_layers)
        ])

        self.W = nn.Linear(hidden_dims, hidden_dims)
        self.V = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, blocks):
        user_embeddings = self.user_embeddings(blocks[0].srcnodes['user'].data[dgl.NID])
        item_embeddings = self.item_embeddings(blocks[0].srcnodes['item'].data[dgl.NID])

        # user_embeddings = user_embeddings + self.U_age(blocks[0].srcnodes['user'].data['age'])
        user_embeddings = user_embeddings + self.U_gender(blocks[0].srcnodes['user'].data['gender'])
        # user_embeddings = user_embeddings + self.U_occupation(blocks[0].srcnodes['user'].data['occupation'])
        item_embeddings = item_embeddings + self.I_genres(blocks[0].srcnodes['item'].data['genres'])


        for block, layer in zip(blocks, self.layers):
            user_embeddings, item_embeddings = layer(block, user_embeddings, item_embeddings)

        user_embeddings = self.W(user_embeddings)
        item_embeddings = self.V(item_embeddings)

        return user_embeddings, item_embeddings

    def compute_score(self, pair_graph, user_embeddings, item_embeddings):
        with pair_graph.local_scope():
            pair_graph.nodes['user'].data['h'] = user_embeddings
            pair_graph.nodes['item'].data['h'] = item_embeddings
            pair_graph.apply_edges(fn.u_dot_v('h', 'h', 'r'))
            return pair_graph.edata['r']

#ml
class GCMCRating2(nn.Module):
    # def __init__(self, num_users, num_items, hidden_dims, num_ratings, num_layers,
    #              num_user_age_bins, num_user_genders, num_user_occupations, num_item_genres):
    def __init__(self, num_users, num_items, hidden_dims, num_ratings, num_layers,
                 num_user_genders, num_item_genres):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, hidden_dims)
        self.item_embeddings = nn.Embedding(num_items, hidden_dims)

        # self.U_age = nn.Embedding(num_user_age_bins, hidden_dims)
        self.U_gender = nn.Embedding(num_user_genders, hidden_dims)
        # self.U_occupation = nn.Embedding(num_user_occupations, hidden_dims)
        self.I_genres = nn.Linear(num_item_genres, hidden_dims)
        self.layers = nn.ModuleList([
            GCMCLayer(hidden_dims, num_ratings) for _ in range(num_layers)
        ])

        self.W = nn.Linear(hidden_dims, hidden_dims)
        self.V = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, blocks):
        user_embeddings = self.user_embeddings(blocks[0].srcnodes['user'].data[dgl.NID])
        item_embeddings = self.item_embeddings(blocks[0].srcnodes['item'].data[dgl.NID])

        # user_embeddings = user_embeddings + self.U_age(blocks[0].srcnodes['user'].data['age'])
        user_embeddings = user_embeddings + self.U_gender(blocks[0].srcnodes['user'].data['gender'])
        # user_embeddings = user_embeddings + self.U_occupation(blocks[0].srcnodes['user'].data['occupation'])
        item_embeddings = item_embeddings + self.I_genres(blocks[0].srcnodes['item'].data['genres'])


        for block, layer in zip(blocks, self.layers):
            user_embeddings, item_embeddings = layer(block, user_embeddings, item_embeddings)

        user_embeddings = self.W(user_embeddings)
        item_embeddings = self.V(item_embeddings)

        return user_embeddings, item_embeddings

    def compute_score(self, pair_graph, user_embeddings, item_embeddings):
        with pair_graph.local_scope():
            pair_graph.nodes['user'].data['h'] = user_embeddings
            pair_graph.nodes['item'].data['h'] = item_embeddings
            pair_graph.apply_edges(fn.u_dot_v('h', 'h', 'r'))
            return pair_graph.edata['r']

def rmse(pred, label):
    return ((pred - label) ** 2).mean().sqrt()
