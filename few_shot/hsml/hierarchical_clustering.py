import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalClustering(nn.Module):
    """
    Hierarchical Clustering for task_emb_vec.
    Structure is a tree-based network (TreeLSTM)
    Input: task_emb_vec [batch_size, task_emb_dim], [1,128]
    Output:  root_node HC-embedded vector [batch_size, hidden_dim], [1,128]

    Attributes:
        num_leaf: 4: number of nodes in the first layer, each node [num_task, hidden_dim] [1, 128]
        num_noleaf: 2: number of nodes in the second layer, each node [num_task, hidden_dim]
        input_dim: task_emb_dim, 128
        hidden_dim: tree hidden_dim, 128
        sigma: for assignment softmax cal, 10.0
        update_nets: dict of update net between different nodes.
        assign_net: AssignNet, which contains clustering centers.
    """
    def __init__(self, args, input_dim, hidden_dim):
        """
        Args:
            :param input_dim: typically 128.
            :param hidden_dim: typically 128.
        """
        super(HierarchicalClustering, self).__init__()
        self.args = args
        self.num_leaf = args.cluster_layer_0
        self.num_noleaf = args.cluster_layer_1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.update_nets = nn.ModuleDict()
        for idx in range(self.num_leaf):   # add update nets
            self.update_nets['update_leaf_{}'.format(idx)] = update_block(input_dim, hidden_dim)
        for jdx in range(self.num_noleaf):
            self.update_nets['update_noleaf_{}'.format(jdx)] = update_block(hidden_dim, hidden_dim)
        self.update_nets['update_root'] = update_block(hidden_dim, hidden_dim)
        self.assign_net = AssignNet(args, self.num_leaf, input_dim, mode='centers')
        if self.num_noleaf > 0:
            self.gate_nets = AssignNet(args, self.num_noleaf, hidden_dim, mode='dense')

        self.apply(self.weight_init)    # customized initialization

    def forward(self, inputs):
        """
        Args:
            :param inputs: [batch_size, task_emb_dim], [1,128]

        Returns:
            :return root_node: HC-embedded vector [batch_size, task_emb_dim], [1,128]
        """
        # layer 0: leaf
        assign = self.assign_net(inputs)    # [4,1]
        leaf_nodes = []     # node values in the first (leaf) layer, [4,1,128]
        for idx in range(self.num_leaf):   # update through the first (leaf) layer
            leaf_nodes.append(assign[idx] * self.update_nets['update_leaf_{}'.format(idx)](inputs))  # [1,128]
        leaf_nodes = torch.stack(leaf_nodes, dim=0)  # [4,1,128]

        # layer 1: noleaf
        if self.num_noleaf > 0:
            noleaf_nodes = []     # node values in the second (noleaf) layer, [2,1,128]
            gates = []      # [4, [2, 1]]
            for idx in range(self.num_leaf):
                gate = self.gate_nets(leaf_nodes[idx])  # [2,1]
                gates.append(gate)
                noleaf_nodes_i = []
                for jdx in range(self.num_noleaf):
                    noleaf_nodes_i.append(gate[jdx] * self.update_nets['update_noleaf_{}'.format(jdx)](leaf_nodes[idx]))
                noleaf_nodes_i = torch.cat(noleaf_nodes_i, dim=0)  # [2,128]
                noleaf_nodes.append(noleaf_nodes_i)
            gates = torch.stack(gates)  # [4, 2, 1]
            noleaf_nodes = torch.stack(noleaf_nodes, dim=0)                 # [4,2,128]
            noleaf_nodes = noleaf_nodes.permute([1, 0, 2])                  # [2,4,128]
            noleaf_nodes = torch.sum(noleaf_nodes, dim=1, keepdim=True)     # [2,1,128]
        else:
            noleaf_nodes = leaf_nodes
            gates = torch.ones((self.num_leaf, 1, 1))

        # layer 2: root
        root_node = []  # node value in the third (root) layer, [1,128]
        num_node = self.num_noleaf if self.num_noleaf > 0 else self.num_leaf
        for jdx in range(num_node):
            root_node.append(self.update_nets['update_root'](noleaf_nodes[jdx]))  # [1,128]
        root_node = torch.stack(root_node, dim=0)  # [2,1,128]
        root_node = torch.sum(root_node, dim=0)  # [1,128]

        return root_node, assign, gates

    @staticmethod
    def weight_init(m):

        if isinstance(m, nn.Linear):    # for update_blocks and gate_net
            nn.init.xavier_uniform_(m.weight)
            nn.init.xavier_uniform_(m.weight)


def update_block(in_channels, out_channels):
    """
    basic update network block. Tanh(Linear(in, out))
    """
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Linear(in_channels, out_channels)),
        # nn.Linear(in_channels, out_channels),
        nn.Tanh(),
    )


class AssignNet(nn.Module):
    """
    Output assignment softmax probability.
    Input: task_emb_vec [batch_size, task_emb_dim], [1,128]
    Output: assign [num_node, batch_size], [4, 1]

    Attributes:
        cluster_centers: [num_node, task_emb_dim], [4,128]
    """
    def __init__(self, args, num_node, input_dim, mode='centers'):
        """
        Args:
            :param num_node: same as number of cluster_center, 4; 2 for gates
            :param input_dim: 128.
            :param mode: 'centers' for cluster_centers approach, 'dense' for dense softmax approach.
        """
        super(AssignNet, self).__init__()
        self.args = args
        self.num_node = num_node
        self.input_dim = input_dim
        self.mode = mode
        if mode == 'centers':
            self.sigma = args.sigma if hasattr(args, 'sigma') else 10.0
            self.cluster_centers = nn.Parameter(torch.randn((num_node, input_dim)))
            nn.init.xavier_uniform_(self.cluster_centers)
            # self.register_parameter('cluster_centers', self.cluster_centers)
        elif mode == 'dense':
            self.denses = nn.ModuleList()
            for idx in range(num_node):     # for second layer, 2 dense gates
                self.denses.append(nn.Linear(input_dim, 1))

    def forward(self, inputs):
        """
        Pass batch of task_emb_vec's to get assignment.

        Args:
            :param inputs: [batch_size, task_emb_dim], [1,128]

        Returns:
            :return assign: assignment probability [num_node, batch_size], [4,1]
        """
        if self.mode == 'centers':
            assign_batch = []   # 4 * [bs, 128]
            for node in range(self.num_node):   # 4
                assign_batch.append(inputs - self.cluster_centers[node])
            assign_batch = torch.stack(assign_batch, dim=0)   # [4,1,128]
            assign = torch.exp(-torch.sum(torch.square(assign_batch), dim=2) / (2.0*self.sigma))   # [4,1]
            assign_sum = torch.sum(assign, dim=0, keepdim=True)     # [1,batch_size], [1,1]
            assign = assign / (assign_sum + 1e-6)    # sum(assign, dim=0) == 1
            return assign

            # assign = torch.stack([inputs for _ in range(self.num_node)], dim=0)  # [4,1,128]
            # assign = assign - torch.stack([self.cluster_centers for _ in range(inputs.shape[0])], dim=1)   # [4,1,128]
            # assign = torch.exp(-torch.sum(torch.square(assign), dim=2) / (2.0*self.sigma))   # [4,1]
            # assign_sum = torch.sum(assign, dim=0, keepdim=True)     # [1,batch_size], [1,1]
            # assign = assign / assign_sum    # sum(assign, dim=0) == 1
            # return assign

        elif self.mode == 'dense':
            assign = []     # num_node*[batch_size,1] 2*[1,1]
            for idx in range(self.num_node):
                dense = self.denses[idx]
                assign.append(dense(inputs))
            assign = torch.cat(assign, dim=1)   # [1,2]
            assign = F.softmax(assign, dim=1)   # i.e., torch.sum(assign,dim=1)==1
            return assign.permute(1, 0)    # [num_node, batch_size], [2,1]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("")
    # ImageEmbedding
    parser.add_argument('--image-height', default=84, type=int)
    parser.add_argument('--image-width', default=84, type=int)
    # HierarchicalClustering
    parser.add_argument('--cluster-layer-0', default=4, type=int)
    parser.add_argument('--cluster-layer-1', default=4, type=int)
    parser.add_argument('--sigma', default=10.0, type=float)
    args = parser.parse_args()

    model = HierarchicalClustering(args, input_dim=128, hidden_dim=128)
    # HierarchicalClustering(
    #   (update_nets): ModuleDict(
    #     (update_leaf_0): Sequential(
    #       (0): Linear(in_features=128, out_features=128, bias=True)
    #       (1): Tanh()
    #     )
    #     (update_leaf_1): Sequential(
    #       (0): Linear(in_features=128, out_features=128, bias=True)
    #       (1): Tanh()
    #     )
    #     (update_leaf_2): Sequential(
    #       (0): Linear(in_features=128, out_features=128, bias=True)
    #       (1): Tanh()
    #     )
    #     (update_leaf_3): Sequential(
    #       (0): Linear(in_features=128, out_features=128, bias=True)
    #       (1): Tanh()
    #     )
    #     (update_noleaf_0): Sequential(
    #       (0): Linear(in_features=128, out_features=128, bias=True)
    #       (1): Tanh()
    #     )
    #     (update_noleaf_1): Sequential(
    #       (0): Linear(in_features=128, out_features=128, bias=True)
    #       (1): Tanh()
    #     )
    #     (update_noleaf_2): Sequential(
    #       (0): Linear(in_features=128, out_features=128, bias=True)
    #       (1): Tanh()
    #     )
    #     (update_noleaf_3): Sequential(
    #       (0): Linear(in_features=128, out_features=128, bias=True)
    #       (1): Tanh()
    #     )
    #     (update_root): Sequential(
    #       (0): Linear(in_features=128, out_features=128, bias=True)
    #       (1): Tanh()
    #     )
    #   )
    #   (assign_net): AssignNet()
    #   (gate_nets): AssignNet(
    #     (denses): ModuleList(
    #       (0): Linear(in_features=128, out_features=1, bias=True)
    #       (1): Linear(in_features=128, out_features=1, bias=True)
    #       (2): Linear(in_features=128, out_features=1, bias=True)
    #       (3): Linear(in_features=128, out_features=1, bias=True)
    #     )
    #   )
    # )
    x = torch.rand(1,128)
    pred = model(x)         # pred {Tensor: {1, 128}, Tensor: {4, 1}, Tensor: {4, 4, 1}}
