import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class TaskSpecific(nn.Module):
    """
    The forward framework for: task embedding -> hierarchical clustering -> {adaptors}.

    Attributes:
        maml_module: basic MAML model to evaluate theta. (used to cal shape of adaptors)
        taskEmbedding: TaskEmbedding model to embed task to get task representation in a latent space.
        hierarchicalClustering: HierarchicalClustering model to get a parameter gate
        which can transform theta into theta_i.
    """
    def __init__(self, args, maml_module, taskEmbedding, hierarchicalClustering):
        """
        Args:
            :param maml_module: basic MAML model to evaluate theta.
            :param taskEmbedding: TaskEmbedding model to embed task to get task representation in a latent space.
            :param hierarchicalClustering: HierarchicalClustering model to get a parameter gate
                which can transform theta into theta_i.
        """
        super(TaskSpecific, self).__init__()
        self.args = args
        self.is_regression = True if hasattr(self.args, "is_regression") and self.args.is_regression else False
        # self.rng = np.random.RandomState(seed=self.args.seed)
        self.maml_module = maml_module
        self.taskEmbedding = taskEmbedding
        self.hierarchicalClustering = hierarchicalClustering
        self.parameter_gate_input_dim = hierarchicalClustering.input_dim + hierarchicalClustering.hidden_dim
        self.adaptors = nn.ModuleDict()     # adaptor for each layer in maml model
        for i, (key, p) in enumerate(self.maml_module.named_parameters()):

            ## if only update the feature extractor, then adaptors should not contain parts for logits.
            if hasattr(self.args, "feature_extractor_only") and self.args.feature_extractor_only:
                if "logits" in key:
                    continue

            weight_size = p.numel()
            self.adaptors['adaptors_{}'.format(i)] = nn.Sequential(
                nn.Linear(self.parameter_gate_input_dim, weight_size), nn.Sigmoid())

    def forward(self, xs_batch, ys_batch, detach_theta0=False, medium_record=False,
                xs_batch_ortho=None, ys_batch_ortho=None):
        """
        Each task goes through: taskEmbedding -> hierarchicalClustering -> parameter gate: theta -> theta_i.

        Args:
            :param xs_batch: support_x batch: [(meta_batch_size+num_cross_samples), N, K, 3, img_size, img_size]
            :param ys_batch: support_y batch: [(meta_batch_size+num_cross_samples), N, Ks]
            :param detach_theta0: true for pool_forward. detach theta_0, that the conflict loss only affect learning cluster net.
        Returns:
            :return task_emb_loss_batch: list of reconstruction loss for the batch
            :return adapted_weights_batch: list of adapted weights (theta_i) for the batch
            :return medium_batch: medium vecs for debug visualization.
        """
        task_emb_loss_batch = []    # since loss in R, batch size [meta_batch_size+num_cross_samples,]
        adapted_weights_batch = []  # list of theta_i

        # medium_batch
        # medium_batch = {'img_emb': [], 'task_emb_vec': [], 'meta_knowledge_h': [], 'etas': [],
        #                 'assign': [], 'assign_for_all': [], 'gates': [], 'task_emb': []}
        medium_batch = {'img_emb': [], 'task_emb_vec': [], 'meta_knowledge_h': [], 'etas': [],
                        'assigns': [], 'gates': []}
        cluster_layer_0 = self.args.cluster_layer_0
        cluster_layer_1 = self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1

        ## -------------
        # Spectral Net
        # process for the whole batch
        if hasattr(self.args, 'use_spectral') and self.args.use_spectral:
            task_emb_vec_batch = []
            for meta_batch in range(xs_batch.shape[0]):

                n, s, c, h, w = xs_batch[meta_batch].shape

                xs = xs_batch[meta_batch].view(-1, c, h, w)
                ys = ys_batch[meta_batch].view(-1)

                # task representation learning
                task_emb_vec, task_emb, task_emb_loss, medium_batch_img_task = \
                    self.taskEmbedding(xs, ys, medium_record=medium_record)
                task_emb_loss_batch.append(task_emb_loss)
                task_emb_vec_batch.append(task_emb_vec)

                if medium_record:
                    medium_batch['img_emb'].append(medium_batch_img_task['img_emb'])     # merge img_emb for each task
                    medium_batch['task_emb_vec'].append(medium_batch_img_task['task_emb_vec'])
                    # medium_batch['task_emb'].append(task_emb.view(n, s, -1).detach().cpu().numpy())
                    # task_emb: list of task embs for all images [batch_size * [N, K, 128]] Tensor

            task_emb_vec_batch = torch.cat(task_emb_vec_batch)        # [bs, 128]

            task_emb_vec_ortho_batch = []
            # task_emb_vec_ortho 用no grad
            with torch.no_grad():
                for meta_batch in range(xs_batch_ortho.shape[0]):

                    n, s, c, h, w = xs_batch_ortho[meta_batch].shape

                    xs = xs_batch_ortho[meta_batch].view(-1, c, h, w)
                    ys = ys_batch_ortho[meta_batch].view(-1)

                    # task representation learning
                    task_emb_vec, _, _, _ = self.taskEmbedding(xs, ys, medium_record=medium_record)
                    task_emb_vec_ortho_batch.append(task_emb_vec)

            task_emb_vec_ortho_batch = torch.cat(task_emb_vec_ortho_batch)        # [bs, 128]
            meta_knowledge_h_batch, spectral_loss = self.hierarchicalClustering(
                task_emb_vec_batch, inputs_ortho=task_emb_vec_ortho_batch)
            task_enhanced_emb_vec_batch = torch.cat([task_emb_vec_batch, meta_knowledge_h_batch], dim=1)  # [bs,256]

            for meta_batch in range(xs_batch.shape[0]):
                meta_knowledge_h = meta_knowledge_h_batch[meta_batch].unsqueeze(0)      # [1, 128]
                task_enhanced_emb_vec = task_enhanced_emb_vec_batch[meta_batch].unsqueeze(0)    # [1, 256]

                if medium_record:
                    medium_batch['meta_knowledge_h'].append(meta_knowledge_h.cpu().detach().numpy())

                # Knowledge Adaptation: FC
                adapted_weights = OrderedDict()
                etas = OrderedDict()
                if detach_theta0:
                    for i, (key, p) in enumerate(self.maml_module.named_parameters()):

                        ## if only update the feature extractor, then adaptors should not contain parts for linear.
                        if hasattr(self.args, "feature_extractor_only") and self.args.feature_extractor_only:
                            if "logits" in key:
                                if "weight" in key:
                                    # p = p.clone().detach().requires_grad_()
                                    # nn.init.xavier_uniform_(p)
                                    p = nn.Parameter(torch.zeros(p.shape)).to(p.device)
                                elif "bias" in key:
                                    p = nn.Parameter(torch.zeros(self.args.num_classes_per_set)).to(p.device)
                                adapted_weights[key] = p
                                continue

                        eta = torch.reshape(self.adaptors['adaptors_{}'.format(i)](task_enhanced_emb_vec),
                                            shape=p.shape)
                        etas[key] = eta.cpu().detach().numpy()
                        # p_detach = p.detach()
                        # p_detach.requires_grad = True
                        # adapted_weights[key] = eta * p_detach
                        adapted_weights[key] = eta * p.detach()  # only things different.
                    adapted_weights_batch.append(adapted_weights)
                else:
                    for i, (key, p) in enumerate(self.maml_module.named_parameters()):

                        ## if only update the feature extractor, then adaptors should not contain parts for linear.
                        if hasattr(self.args, "feature_extractor_only") and self.args.feature_extractor_only:
                            if "logits" in key:
                                if "weight" in key:
                                    # p = p.clone().detach().requires_grad_()
                                    # nn.init.xavier_uniform_(p)
                                    p = nn.Parameter(torch.zeros(p.shape)).to(p.device)
                                elif "bias" in key:
                                    p = nn.Parameter(torch.zeros(self.args.num_classes_per_set)).to(p.device)
                                adapted_weights[key] = p
                                continue

                        eta = torch.reshape(self.adaptors['adaptors_{}'.format(i)](task_enhanced_emb_vec),
                                            shape=p.shape)
                        etas[key] = eta.cpu().detach().numpy()
                        adapted_weights[key] = eta * p
                    adapted_weights_batch.append(adapted_weights)

                if medium_record:
                    medium_batch['etas'].append(etas)

            task_emb_loss_batch = torch.stack(task_emb_loss_batch)

            return task_emb_loss_batch, adapted_weights_batch, medium_batch, spectral_loss
        # Spectral Net
        ## -------------

        for meta_batch in range(xs_batch.shape[0]):     # [bs, n, s/kq, (c,h,w)/2]

            n, s = xs_batch[meta_batch].shape[:2]
            input_dim = xs_batch[meta_batch].shape[2:]      # (1,) for regression, (c, h, w) for classification

            xs = xs_batch[meta_batch].view((-1,) + input_dim)
            ys = ys_batch[meta_batch].view(-1)

            # task representation learning
            task_emb_vec, task_emb, task_emb_loss, medium_batch_img_task = \
                self.taskEmbedding(xs, ys, medium_record=medium_record)
            task_emb_loss_batch.append(task_emb_loss)

            if medium_record:
                medium_batch['img_emb'].append(medium_batch_img_task['img_emb'])     # merge img_emb for each task
                medium_batch['task_emb_vec'].append(medium_batch_img_task['task_emb_vec'])
                # medium_batch['task_emb'].append(task_emb.view(n, s, -1).detach().cpu().numpy())
                # task_emb: list of task embs for all images [batch_size * [N, K, 128]] Tensor

            # Hierarchical Task Clustering
            meta_knowledge_h, assign, gates = self.hierarchicalClustering(task_emb_vec)  # [1,128]
            task_enhanced_emb_vec = torch.cat([task_emb_vec, meta_knowledge_h], dim=1)  # [1,256]

            # cal assign_for_all    task_emb: [NK,128]
            # assign_for_all = self.hierarchicalClustering.assign_net(torch.mean(task_emb.view(n, s, -1), dim=1)) # [4, N]

            if medium_record:
                medium_batch['meta_knowledge_h'].append(meta_knowledge_h.cpu().detach().numpy())
                medium_batch['assigns'].append(assign.view(cluster_layer_0).detach().cpu().numpy())    # 4(bs) * [4]
                # medium_batch['assign_for_all'].append(assign_for_all)    # 4(bs) * [4, 5]
                medium_batch['gates'].append(gates.view(cluster_layer_0, cluster_layer_1).detach().cpu().numpy())      # 4(bs) * [4, 2]

            # Knowledge Adaptation: FC
            adapted_weights = OrderedDict()
            etas = OrderedDict()
            if detach_theta0:
                for i, (key, p) in enumerate(self.maml_module.named_parameters()):

                    ## if only update the feature extractor, then adaptors should not contain parts for linear.
                    if hasattr(self.args, "feature_extractor_only") and self.args.feature_extractor_only:
                        if "logits" in key:
                            if "weight" in key:
                                # p = p.clone().detach().requires_grad_()
                                # nn.init.xavier_uniform_(p)
                                p = nn.Parameter(torch.zeros(p.shape)).to(p.device)
                            elif "bias" in key:
                                p = nn.Parameter(torch.zeros(self.args.num_classes_per_set)).to(p.device)
                            adapted_weights[key] = p
                            continue

                    eta = torch.reshape(self.adaptors['adaptors_{}'.format(i)](task_enhanced_emb_vec), shape=p.shape)
                    etas[key] = eta.cpu().detach().numpy()
                    # p_detach = p.detach()
                    # p_detach.requires_grad = True
                    # adapted_weights[key] = eta * p_detach
                    adapted_weights[key] = eta * p.detach()         # only things different.
                adapted_weights_batch.append(adapted_weights)
            else:
                for i, (key, p) in enumerate(self.maml_module.named_parameters()):

                    ## if only update the feature extractor, then adaptors should not contain parts for linear.
                    if hasattr(self.args, "feature_extractor_only") and self.args.feature_extractor_only:
                        if "logits" in key:
                            if "weight" in key:
                                # p = p.clone().detach().requires_grad_()
                                # nn.init.xavier_uniform_(p)
                                p = nn.Parameter(torch.zeros(p.shape)).to(p.device)
                            elif "bias" in key:
                                p = nn.Parameter(torch.zeros(self.args.num_classes_per_set)).to(p.device)
                            adapted_weights[key] = p
                            continue

                    eta = torch.reshape(self.adaptors['adaptors_{}'.format(i)](task_enhanced_emb_vec), shape=p.shape)
                    etas[key] = eta.cpu().detach().numpy()
                    adapted_weights[key] = eta * p
                adapted_weights_batch.append(adapted_weights)

            if medium_record:
                medium_batch['etas'].append(etas)

        # [meta_batch_size+num_cross_samples,]
        if len(task_emb_loss_batch) is not 0:       # 0 if no mix-task used
            task_emb_loss_batch = torch.stack(task_emb_loss_batch)

        return task_emb_loss_batch, adapted_weights_batch, medium_batch

    def forward_get_task_emb(self, xs_batch, ys_batch):
        """
        get the corresponding task_emb  # [bs * [N, K, 128]]
        Args:
            xs_batch:  [bs, N, K, 3, img_size, img_size]
            ys_batch:  [bs, N, K]

        Returns: task_embs  [bs * [N, K, 128]]
        """
        task_embs = list()
        for meta_batch in range(xs_batch.shape[0]):

            if self.is_regression:
                n, s, input_dim = xs_batch[meta_batch].shape

                xs = xs_batch[meta_batch].view(-1, input_dim)
            else:
                n, s, c, h, w = xs_batch[meta_batch].shape

                xs = xs_batch[meta_batch].view(-1, c, h, w)

            ys = ys_batch[meta_batch].view(-1)

            # task representation learning
            task_emb_vec, task_emb, _, _ = self.taskEmbedding(xs, ys)

            task_embs.append(task_emb.view(n, s, -1))

        return task_embs

    def forward_task_emb(self, task_embs):
        """
        obtain theta_is for the given task_emb
        used for mixup_representation
        Note that, the theta_0 is detached that this the gradient will not backward to theta_0
        Args:
            task_embs: [bs, 128]
        Returns: adapted_weights_batch: [bs, ...] list of adapted_weights

        """
        adapted_weights_batch = []  # list of theta_i

        for task_emb in task_embs:      # [4,128] -> [128,]
            task_emb = task_emb.unsqueeze(0)      # [128,] -> [1,128]
            # Hierarchical Task Clustering
            meta_knowledge_h, assign, gates = self.hierarchicalClustering(task_emb)  # [1,128]
            task_enhanced_emb_vec = torch.cat([task_emb, meta_knowledge_h], dim=1)  # [1,256]

            # Knowledge Adaptation: FC
            adapted_weights = OrderedDict()
            etas = OrderedDict()
            for i, (key, p) in enumerate(self.maml_module.named_parameters()):

                ## if only update the feature extractor, then adaptors should not contain parts for linear.
                if hasattr(self.args, "feature_extractor_only") and self.args.feature_extractor_only:
                    if "linear" in key:
                        if "weight" in key:
                            # p = p.clone().detach().requires_grad_()
                            # nn.init.xavier_uniform_(p)
                            p = nn.Parameter(torch.zeros(p.shape)).to(p.device)
                        elif "bias" in key:
                            p = nn.Parameter(torch.zeros(self.args.num_classes_per_set)).to(p.device)
                        adapted_weights[key] = p       # 这里是给classifier置0，不用detach
                        continue

                eta = torch.reshape(self.adaptors['adaptors_{}'.format(i)](task_enhanced_emb_vec), shape=p.shape)
                etas[key] = eta.cpu().detach().numpy()
                adapted_weights[key] = eta * p.detach()         # prevent gradient backward to theta_0 (p)
            adapted_weights_batch.append(adapted_weights)

        return adapted_weights_batch

    @torch.no_grad()
    def forward_task_emb_get_clustering_information(self, task_emb_vecs):
        """

        Args:
            task_emb_vecs:  [bs, vec_dim]

        Returns: np
            assigns: [bs, cluster_layer_0]
            gates: [bs, cluster_layer_0, cluster_layer_1]
            cluster_idxs: [bs]
            scores: [bs, cluster_layer_0 * cluster_layer_1]
        """
        bs, vec_dim = task_emb_vecs.shape
        # cal clustering information for each vec [1, 128]
        assigns = []
        gates = []
        cluster_idxs = []
        scores = []
        for task_id in range(bs):
            task_emb_vec = task_emb_vecs[task_id].view(1, -1)  # [1, 128]
            _, assign, gate = self.hierarchicalClustering(task_emb_vec)
            assigns.append(assign.view(self.args.cluster_layer_0).cpu().detach().numpy())
            gates.append(gate.view(self.args.cluster_layer_0, self.args.cluster_layer_1).cpu().detach().numpy())
            # calculate the score [cluster_layer_0*cluster_layer_1, ]
            score = []
            for idx1 in range(self.args.cluster_layer_0):
                for idx2 in range(self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1):
                    score.append(assigns[task_id][idx1] * gates[task_id][idx1][idx2])
            scores.append(score)
        assigns = np.stack(assigns)     # [bs, 4]
        gates = np.stack(gates)         # [bs, 4, 2]
        scores = np.array(scores)       # [bs, 8]
        assert np.sum(assigns[0]) > 0.98
        assert np.sum(gates[0][0]) > 0.98
        assert np.sum(scores[0]) > 0.98

        # cluster_idxs
        cluster_idxs = np.argmax(scores, axis=1)   # [bs]

        return assigns, gates, cluster_idxs, scores

    @torch.no_grad()
    def forward_get_clustering_information(self, xs_batch, ys_batch):
        """

        Args:
            xs_batch:  [bs, N, K, 3, img_size, img_size]
            ys_batch:  [bs, N, K]

        Returns: np
            assigns: [bs, cluster_layer_0]
            gates: [bs, cluster_layer_0, cluster_layer_1]
            cluster_idxs: [bs]
            scores: [bs, cluster_layer_0 * cluster_layer_1]
        """
        task_embs = self.forward_get_task_emb(xs_batch, ys_batch)   # [bs * [N, K, 128]]

        # obtain all task_emb_vecs [1, 128]
        task_embs = torch.stack(task_embs)    # [batch_size, N, K, 128]
        bs, n, k, vec_dim = task_embs.shape
        task_emb_vecs = torch.mean(task_embs.view(bs, -1, vec_dim), dim=1)    # [batch_size, 128]

        return self.forward_task_emb_get_clustering_information(task_emb_vecs)

    def structurize_clustering_information(self, assigns, gates):
        """
        assigns, gates -> assigns, gates, cluster_idxs, scores
        Args:
            assigns:
            gates:

        Returns: np
            assigns: [bs, cluster_layer_0]
            gates: [bs, cluster_layer_0, cluster_layer_1]
            cluster_idxs: [bs]
            scores: [bs, cluster_layer_0 * cluster_layer_1]

        """
        assigns = np.array(assigns)
        gates = np.array(gates)
        bs, cluster_layer_0 = assigns.shape
        _, _, cluster_layer_1 = gates.shape

        cluster_idxs = []
        scores = []
        for task_id in range(bs):
            score = []
            for idx1 in range(cluster_layer_0):
                for idx2 in range(cluster_layer_1):
                    score.append(assigns[task_id][idx1] * gates[task_id][idx1][idx2])
            scores.append(score)
        scores = np.array(scores)       # [bs, 8]
        assert np.sum(assigns[0]) > 0.98
        assert np.sum(gates[0][0]) > 0.98
        assert np.sum(scores[0]) > 0.98

        # cluster_idxs
        cluster_idxs = np.argmax(scores, axis=1)   # [bs]

        return assigns, gates, cluster_idxs, scores


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
    # TaskEmbedding
    parser.add_argument('--is-regression', default=False, type=bool)
    parser.add_argument('--ae-type', default='gru', type=str)
    parser.add_argument('--num-classes-per-set', default=5, type=int)
    # TaskSpecific
    parser.add_argument('--feature-extractor-only', default=False, type=bool)
    parser.add_argument('--use-spectral', default=False, type=bool)
    args = parser.parse_args()

    from few_shot.models import FewShotClassifier
    from few_shot.hsml.task_embedding import TaskEmbedding
    from few_shot.hsml.hierarchical_clustering import HierarchicalClustering

    maml_module = FewShotClassifier(num_input_channels=3, k_way=args.num_classes_per_set,
                                    final_layer_size=1200, number_filters=48)
    taskEmbedding = TaskEmbedding(args, task_emb_dim=128)
    hierarchicalClustering = HierarchicalClustering(args, input_dim=128, hidden_dim=128)
    model = TaskSpecific(args, maml_module, taskEmbedding, hierarchicalClustering)

    x = torch.rand(1,5,1,3,84,84)       # 1 task
    y = torch.tensor([0,1,2,3,4]).unsqueeze(0).unsqueeze(-1)
    pred = model(x, y, detach_theta0=True, medium_record=True)
    # pred {Tensor: {1, 128}, Tensor: {5, 128}, Tensor: {},
    #       {'img_emb': ndarray {5, 64}, 'task_emb_vec': ndarray {1, 128}}}
