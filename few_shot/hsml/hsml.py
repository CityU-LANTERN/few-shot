import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Callable, Union

from few_shot.models import FewShotClassifier
from few_shot.hsml.task_embedding import TaskEmbedding
from few_shot.hsml.hierarchical_clustering import HierarchicalClustering
from few_shot.hsml.task_specific import TaskSpecific
from few_shot.core import create_nshot_task_label
from few_shot.maml import meta_gradient_step

from pymoo.factory import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from sklearn.cluster import KMeans

import queue
from matplotlib import pyplot as plt


class HSML(nn.Module):
    """
    HSML maintain a clustering pool for classes not for tasks
    Attributes:
        maml_module: basic MAML model to evaluate theta.
        taskEmbedding: TaskEmbedding model to embed task to get task representation in a latent space.
        hierarchicalClustering: HierarchicalClustering model to get a parameter gate
        which can transform theta into theta_i.
    """
    def __init__(self, args, device):
        """
        HSML few-shot classification system
        :param device: The device to use
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(HSML, self).__init__()
        self.args = args
        self.is_regression = True if hasattr(self.args, "is_regression") and self.args.is_regression else False
        self.batch_size = args.batch_size
        self.current_epoch = 0
        self.trained_iterations = 0

        self.y_support_set = np.array([
            [c_idx for _ in range(self.args.num_samples_per_class)] for c_idx in range(self.args.num_classes_per_set)
        ])  # [n, s]
        self.y_target_set = np.array([
            [c_idx for _ in range(self.args.num_target_samples)] for c_idx in range(self.args.num_classes_per_set)
        ])  # [n, kq]
        self.q_relative_labels = create_nshot_task_label(k=self.args.num_classes_per_set, q=self.args.num_target_samples).to(device)

        self.pool_rng = np.random.RandomState(seed=42)
        base_learner = FewShotClassifier(num_input_channels=args.num_input_channels, k_way=args.num_classes_per_set,
                                         final_layer_size=args.final_layer_size, number_filters=args.number_filters)

        # if hasattr(self.args, "detach_theta0") and self.args.detach_theta0:
        #     for name, param in self.named_parameters():
        #         param.requires_grad = False

        taskEmbedding = TaskEmbedding(args, args.hidden_dim)
        hierarchical_clustering = HierarchicalClustering(args, args.hidden_dim, args.hidden_dim)
        self.taskSpecificNet = TaskSpecific(args, base_learner, taskEmbedding, hierarchical_clustering)

        self.pool_size = self.args.pool_size if hasattr(
            self.args, "pool_size"
        ) else self.args.cluster_layer_0 * (self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1)

        # self.pool = [[] for _ in range(pool_size)]
        # self.val_pool = [[] for _ in range(pool_size)]
        self.val_iter = 0
        self.pool = self.init_pool()
        self.val_pool = self.init_pool()
        # used in val_iteration, auto-increase when cal val_iteration(),
        # (val_iter+1) % int(num_evaluation_tasks / batch_size) == 0 means this batch is the last batch in this epoch.

        # store the item (tuple):
        #   ['class': class sample in task; only x_support_set and x_query_set for 1 class
        #    'assign': assign value to assign this class to this C]
        # it is no need to store y, since they are relative label with 1-5
        # in CPU memory
        self.pool_max_size = self.args.pool_max_size if hasattr(
            self.args, 'pool_max_size') else self.args.num_classes_per_set*4

        self.device = device

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def init_pool(self):
        return {'pool': [[] for _ in range(self.pool_size)], 'centers': None}

    def get_accross_indi_loss_metrics(self, losses_list):
        losses = dict()
        for idx, indi_losses in enumerate(losses_list):
            for key, item in indi_losses.items():
                losses['{}_{}'.format(idx, key)] = item     # 0_loss or 1_loss_0

        # if loss for each indi is distributed across cuda devices, this losses['loss'] is not used to meta_update
        losses['loss'] = np.mean(np.stack([losses['{}_loss'.format(idx)].cpu().detach().numpy()
                                           for idx in range(len(losses_list))]))
        losses['accuracy'] = np.mean(np.stack([losses['{}_accuracy'.format(idx)] for idx in range(len(losses_list))]))

        return losses

    @torch.no_grad()
    def class_batch_2_class_embs(self, class_batch, n):
        """
        Args:
            class_batch: cls sup x, cls tgt x
            n: num_classes_per_set 5
        Returns:
            class_embs: [bs, s+kq, N, K, 128]
            class_emb_vecs: [bs, 1, 128]
        """
        x_support_set, x_target_set, y_support_set, y_target_set = class_batch
        # [batch_size, shot/Kq, c, h, w]; [batch_size, shot]
        # class: [shot/Kq, c, h, w]; [shot/Kq]

        if self.is_regression:
            bs, s, input_dim = x_support_set.shape
            _, kq, _ = x_target_set.shape
        else:
            bs, s, c, h, w = x_support_set.shape
            _, kq, _, _, _ = x_target_set.shape

        class_embs = []
        class_emb_vecs = []
        # for each class instance   bs*N
        for cls_idx in range(bs):
            x_support_set_dup = []
            y_support_set_dup = []
            # concat all img in the support set and query set
            img_set = torch.cat([x_support_set[cls_idx], x_target_set[cls_idx]])    # [s+kq, c,h,w]
            lab_set = torch.cat([y_support_set[cls_idx], y_target_set[cls_idx]])
            # for each img
            for img_idx in range(img_set.shape[0]):     # s+kq
                img = img_set[img_idx]     # [c, h, w]
                lab = lab_set[img_idx]
                # duplicate img KN times to form a support set as pure task
                x_support_set_dup.append(torch.stack([
                    torch.stack([img for _ in range(s)]) for _ in range(n)
                ]))
                y_support_set_dup.append(torch.stack([
                    torch.stack([lab for _ in range(s)]) for _ in range(n)
                ]))
            x_support_set_dup = torch.stack(x_support_set_dup)  # [(s+kq), n, s, c, h, w]
            y_support_set_dup = torch.stack(y_support_set_dup)

            if not self.is_regression:      # classification, use relative label
                y_support_set_dup = np.stack([self.y_support_set for _ in range(s+kq)])  # [(s+kq), n, s]

            assert y_support_set_dup.shape == (s+kq, n, s)

            x_support_set_dup = x_support_set_dup.double().to(device=self.device)
            if self.is_regression:
                y_support_set_dup = torch.Tensor(y_support_set_dup).double().to(device=self.device)
            else:
                y_support_set_dup = torch.Tensor(y_support_set_dup).long().to(device=self.device)

            task_embs = self.taskSpecificNet.forward_get_task_emb(x_support_set_dup, y_support_set_dup)
            task_embs = torch.stack(task_embs)  # [s+kq, N, K, 128]
            _, _, _, vec_dim = task_embs.shape
            # task_emb_vec for each class instance as average of all (s+kq)*N*K [1, 128]
            task_emb_vec = torch.mean(task_embs.view(-1, vec_dim), dim=0, keepdim=True)  # [1, 128]
            class_embs.append(task_embs)
            class_emb_vecs.append(task_emb_vec)

        class_embs = torch.stack(class_embs)        # [bs, s+kq, N, K, 128]
        class_emb_vecs = torch.stack(class_emb_vecs)    # [bs, 1, 128]

        return class_embs, class_emb_vecs

    @torch.no_grad()
    def pool_put_argmax(self, task_batch, assigns, gates, pool=None):
        """
        pool_put 是把class放到pool里
        maintain the pool. add a batch of classes from task in task_batch into the pool
        when put, first check if qsize() >= self.pool_max_size remove one by get()
        Note that, task_batch should be in CPU, to decrease GPU memory usage.
        Args:
            pool: the target pool
            task_batch: batch of tasks
            assigns: [batch_size, N, 4]
            gates: [batch_size, N, 4, 2]
        Returns:
        """
        if pool is None:
            pool = self.pool

        x_support_set, x_target_set, y_support_set, y_target_set = task_batch
        # [batch_size, n_way, shot/Kq, c, h, w]; [batch_size, n_way, shot/Kq]
        # class: [shot/Kq, c, h, w]; [shot/Kq]

        assigns = assigns.reshape(-1, self.args.cluster_layer_0)
        # [batch_size*N, 4]
        gates = gates.reshape(-1, self.args.cluster_layer_0, self.args.cluster_layer_1)
        # [batch_size*N, 4, 2, 1]

        cluster_idxs, scores = self.predict_clusters_with_gate(assigns, gates)  # [batch_size*N]; [batch_size*N]
        cluster_idxs = cluster_idxs.reshape(x_support_set.shape[0], x_support_set.shape[1])
        scores = scores.reshape(x_support_set.shape[0], x_support_set.shape[1], -1)

        for task_idx in range(x_support_set.shape[0]):
            for n_idx in range(x_support_set.shape[1]):
                cluster_idx = cluster_idxs[task_idx, n_idx]
                c = (x_support_set[task_idx, n_idx], x_target_set[task_idx, n_idx],
                     y_support_set[task_idx, n_idx], y_target_set[task_idx, n_idx])
                pool[cluster_idx].append({'class': c, 'assign': scores[task_idx, n_idx, cluster_idx]})
                # score is assign*gate
                self.pop_class(cluster_idx, pool)

    @torch.no_grad()
    def pool_put_all(self, task_batch, true_labels, scores, pool=None):
        """
        pool_put 是把class放到pool里. 每个column都放
        maintain the pool. add a batch of classes from task in task_batch into the pool
        when put, first check if qsize() >= self.pool_max_size remove one by get()
        Note that, task_batch should be in CPU, to decrease GPU memory usage.
        Args:
            pool: the target pool
            task_batch: batch of tasks
            true_labels: [batch_size, N]
            scores: [batch_size, N, 4*2]
        Returns:
        """
        if pool is None:
            pool = self.pool

        assert type(pool) is list

        x_support_set, x_target_set, _, _ = task_batch
        # [batch_size, n_way, shot/Kq, c, h, w]; [batch_size, n_way, shot/Kq]
        # class: [shot/Kq, c, h, w]; [shot/Kq]

        assert len(true_labels) == x_support_set.shape[0]
        assert len(true_labels[0]) == x_support_set.shape[1]

        cluster_layer_0 = self.args.cluster_layer_0
        cluster_layer_1 = self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1

        for task_idx in range(x_support_set.shape[0]):
            for n_idx in range(x_support_set.shape[1]):
                c = (x_support_set[task_idx, n_idx], x_target_set[task_idx, n_idx])
                # put c into all columns
                for cluster_idx in range(cluster_layer_0 * cluster_layer_1):
                    label_list = np.array([item['label'] for item in pool[cluster_idx]])
                    assign_list = np.array([item['assign'] for item in pool[cluster_idx]])
                    assign_idx_in_pool = np.where(label_list == true_labels[task_idx][n_idx])[0]
                    assign_in_pool = assign_list[assign_idx_in_pool]

                    # print('---------debug----------')
                    # print("label_list:", label_list)
                    # print("assign_list:", assign_list)
                    # print("assign_idx_in_pool:", assign_idx_in_pool)
                    # print("assign_in_pool:", assign_in_pool)
                    # results:
                    # label_list: ['Cessna-208' 'DC-9-30' 'Challenger-600' 'BAE-146-300' 'EMB-120' 'DH-82'
                    #  '171.Myrtle_Warbler' '181.Worm_eating_Warbler' 'DHC-8-300'
                    #  '113.Baird_Sparrow']
                    # assign_list: [0.12746418 0.12745473 0.12747617 0.12750468 0.12753595 0.1274124
                    #  0.12761356 0.12749223 0.12742798 0.12742291]
                    # assign_idx_in_pool: []
                    # assign_in_pool: []

                    # if pool[cluster_idx] contain this label, just compare this 2 c's score
                    if len(assign_in_pool) > 0 and assign_in_pool[0] < scores[task_idx, n_idx, cluster_idx]:
                        pool[cluster_idx].pop(assign_idx_in_pool[0])
                        pool[cluster_idx].append({'class': c,
                                                  'assign': scores[task_idx, n_idx, cluster_idx],
                                                  'label': true_labels[task_idx][n_idx]})
                    # else put and pop 1 c with smallest score.
                    elif len(assign_in_pool) == 0:
                        pool[cluster_idx].append({'class': c,
                                                  'assign': scores[task_idx, n_idx, cluster_idx],
                                                  'label': true_labels[task_idx][n_idx]})
                        # delete the smallest one to maintain pool_max_size items.
                        if len(pool[cluster_idx]) == self.pool_max_size + 1:
                            pool[cluster_idx].pop(
                                int(np.argmin(np.array([item['assign'] for item in pool[cluster_idx]]))))

    @torch.no_grad()
    def pool_put_kmeans(self, task_batch, true_labels, scores, pool=None, reduce=False, flag='during'):
        """
        using the current batch's class scores and the population's scores to update the 8 cluster centers
        for val, just put new class instances into val_pool according to the trained centers
        Args:
            pool: the target pool
            task_batch: batch of tasks
            true_labels: [batch_size, N]
            scores: [batch_size, N, 4*2]
            reduce: False to not reduce the number and consider true labels; true to do so.
        Returns:
        """
        if pool is None:
            pool = self.pool

        assert 'pool' in pool.keys() and 'centers' in pool.keys()

        x_support_set, x_target_set, y_support_set, y_target_set = task_batch
        # [batch_size, n_way, shot/Kq, c, h, w]; [batch_size, n_way, shot/Kq]
        # class: [shot/Kq, c, h, w]; [shot/Kq]

        assert len(true_labels) == x_support_set.shape[0]
        assert len(true_labels[0]) == x_support_set.shape[1]

        cluster_layer_0 = self.args.cluster_layer_0
        cluster_layer_1 = self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1
        dim = self.pool_size

        # get population for new instances 20 * {class: (spt,que), score: [8], label: str}
        pop = []
        for task_idx in range(x_support_set.shape[0]):
            for n_idx in range(x_support_set.shape[1]):
                c = (x_support_set[task_idx, n_idx], x_target_set[task_idx, n_idx],
                     y_support_set[task_idx, n_idx], y_target_set[task_idx, n_idx])
                pop.append({'class': c, 'score': scores[task_idx, n_idx], 'label': true_labels[task_idx][n_idx]})

        if pool is self.val_pool and "_medium" in flag:
            # just randomly put class instances in pop into pool.
            for pop_idx, indi in enumerate(pop):
                column_idx = self.pool_rng.randint(dim)
                dist = 0

                # put indi to column_idx's column
                if not reduce:      # just put into the pool
                    pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                     'label': indi['label'],
                                                     'dist': dist})
                else:
                    same_label_idxs = []
                    for idx, item in enumerate(pool['pool'][column_idx]):
                        if item['label'] == indi['label']:
                            same_label_idxs.append(idx)
                    dist_list = np.array([item['dist'] for item in pool['pool'][column_idx]])

                    # if pool['pool'][column_idx] contain this label, just compare this 2 c's dist
                    if len(same_label_idxs) > 0 and dist_list[same_label_idxs][0] > dist:       # should be only 1 since put 1 by 1.
                        pool['pool'][column_idx].pop(same_label_idxs[0])
                        pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                         'label': indi['label'],
                                                         'dist': dist})
                    # else just put.
                    elif len(same_label_idxs) == 0:
                        pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                         'label': indi['label'],
                                                         'dist': dist})

                        # delete the largest dist one to maintain pool_max_size items.
                        if len(pool['pool'][column_idx]) == self.pool_max_size + 1:
                            pool['pool'][column_idx].pop(
                                int(np.argmax(np.array([item['dist'] for item in pool['pool'][column_idx]]))))

        # if pool is self.pool or (pool is self.val_pool and flag is 'after'):
        # elif pool is self.pool or pool is self.val_pool:
        else:
            # apply kmeans for new class instances and instances in the pool
            for column in pool['pool']:
                for indi in column:
                    pop.append({'class': indi['class'], 'score': indi['score'], 'label': indi['label']})
            pool['pool'] = [[] for _ in range(dim)]

            # apply sklearn kmeans
            score_data = np.stack([item['score'] for item in pop])
            km = KMeans(n_clusters=dim, init='k-means++', random_state=self.pool_rng)   # , max_iter=50
            column_idxs = km.fit_predict(score_data)     # [len(pop)]
            centroids = km.cluster_centers_
            pool['centers'] = centroids

            # # ------debug-----------
            # print('centroids: \n', centroids)
            # # ------debug-----------

            # put each class instance into the pool
            for pop_idx, indi in enumerate(pop):
                column_idx = column_idxs[pop_idx]
                dist = np.sqrt(np.sum(np.square(score_data[pop_idx] - centroids[column_idx])))

                # put indi to column_idx's column
                if not reduce:      # just put into the pool
                    pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                     'label': indi['label'],
                                                     'dist': dist})
                else:
                    if pool is self.pool:
                        same_label_idxs = []
                        for idx, item in enumerate(pool['pool'][column_idx]):
                            if item['label'] == indi['label']:
                                same_label_idxs.append(idx)
                        dist_list = np.array([item['dist'] for item in pool['pool'][column_idx]])

                        # if pool['pool'][column_idx] contain this label, just compare this 2 c's dist
                        if len(same_label_idxs) > 0 and dist_list[same_label_idxs][0] > dist:       # should be only 1 since put 1 by 1.
                            pool['pool'][column_idx].pop(same_label_idxs[0])
                            pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                             'label': indi['label'],
                                                             'dist': dist})
                        # else just put.
                        elif len(same_label_idxs) == 0:
                            pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                             'label': indi['label'],
                                                             'dist': dist})

                            # delete the largest dist one to maintain pool_max_size items.
                            if len(pool['pool'][column_idx]) == self.pool_max_size + 1:
                                pool['pool'][column_idx].pop(
                                    int(np.argmax(np.array([item['dist'] for item in pool['pool'][column_idx]]))))

                    # for val, do not hold only-1-instance-for-1-class constraint.
                    elif pool is self.val_pool:
                        pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                         'label': indi['label'],
                                                         'dist': dist})

                        # delete the largest dist one to maintain pool_max_size items.
                        if len(pool['pool'][column_idx]) == self.pool_max_size + 1:
                            pool['pool'][column_idx].pop(
                                int(np.argmax(np.array([item['dist'] for item in pool['pool'][column_idx]]))))

            # # if cluster_centers are None, randomly choose 8 from population.
            # if pool['centers'] is None:
            #     # select dim class instances' score as the cluster_centers
            #     available_idx = np.arange(len(pop))
            #     selected_idx = self.pool_rng.choice(available_idx, size=dim, replace=False)
            #     pool['centers'] = np.stack([pop[idx]['score'] for idx in selected_idx])      # [8, 8]
            #
            # for kmeans_iter in range(9):
            #     self.pool_assign(pop, pool, reduce=False)
            #     self.pool_center_update(pool)
            #
            # self.pool_assign(pop, pool, reduce=reduce)
            # self.pool_center_update(pool)

        # # just put new class instances into val_pool according to the trained centers
        # elif pool is self.val_pool and flag is 'during':
        #
        #     centers = self.pool['centers']
        #     for pop_idx, indi in enumerate(pop):
        #         dists = np.sqrt(np.sum((centers - indi['score']) ** 2, axis=1))
        #         column_idx = np.argmin(dists)
        #
        #         # put indi to column_idx's column
        #         if not reduce:      # just put into the pool
        #             pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
        #                                              'label': indi['label'],
        #                                              'dist': dists[column_idx]})
        #         else:
        #             same_label_idxs = []
        #             for idx, item in enumerate(pool['pool'][column_idx]):
        #                 if item['label'] == indi['label']:
        #                     same_label_idxs.append(idx)
        #             dist_list = np.array([item['dist'] for item in pool['pool'][column_idx]])
        #
        #             # if pool['pool'][column_idx] contain this label, just compare this 2 c's dist
        #             if len(same_label_idxs) > 0 and dist_list[same_label_idxs][0] > dists[column_idx]:
        #                 pool['pool'][column_idx].pop(same_label_idxs[0])
        #                 pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
        #                                                  'label': indi['label'],
        #                                                  'dist': dists[column_idx]})
        #
        #             # else just put.
        #             elif len(same_label_idxs) == 0:
        #                 pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
        #                                                  'label': indi['label'],
        #                                                  'dist': dists[column_idx]})
        #
        #                 # delete the largest dist one to maintain pool_max_size items.
        #                 if len(pool['pool'][column_idx]) == self.pool_max_size + 1:
        #                     pool['pool'][column_idx].pop(
        #                         int(np.argmax(np.array([item['dist'] for item in pool['pool'][column_idx]]))))

        # sort each column in an ascending distance order.
        for cl_idx, column in enumerate(pool['pool']):
            dist_list = np.array([item['dist'] for item in column])
            sorted_idxs = np.argsort(dist_list)
            pool['pool'][cl_idx] = [column[idx] for idx in sorted_idxs]

    @torch.no_grad()
    def pool_put_kmeans_cheat(self, task_batch, true_labels, scores, dataset_idxs, pool=None, reduce=False):
        """
        using the current batch's class scores and the population's scores to update the 8 cluster centers
        for val, just put new class instances into val_pool according to the trained centers
        Args:
            pool: the target pool
            task_batch: batch of tasks
            true_labels: [batch_size, N]
            scores: [batch_size, N, 4*2]
            reduce: False to not reduce the number and consider true labels; true to do so.
        Returns:
        """
        if pool is None:
            pool = self.pool

        assert 'pool' in pool.keys() and 'centers' in pool.keys()

        x_support_set, x_target_set, y_support_set, y_target_set = task_batch
        # [batch_size, n_way, shot/Kq, c, h, w]; [batch_size, n_way, shot/Kq]
        # class: [shot/Kq, c, h, w]; [shot/Kq]

        assert len(true_labels) == x_support_set.shape[0]
        assert len(true_labels[0]) == x_support_set.shape[1]

        cluster_layer_0 = self.args.cluster_layer_0
        cluster_layer_1 = self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1
        score_dim = cluster_layer_0 * cluster_layer_1
        dim = self.pool_size

        # set(dataset_idxs) and separately process for each dataset.
        dataset_idxs_set = [i for i in range(len(self.args.dataset_name))]       # [0,1]
        num_columns_per_dataset = int(dim / len(dataset_idxs_set))          # 4
        assert num_columns_per_dataset * len(dataset_idxs_set) == dim
        pop = dict()
        for dataset_idx in dataset_idxs_set:
            pop[dataset_idx] = []

        # get population (20 + 80) * {class: (spt,que), score: [8], label: str}
        for task_idx in range(x_support_set.shape[0]):
            dataset_idx = dataset_idxs[task_idx]
            for n_idx in range(x_support_set.shape[1]):
                c = (x_support_set[task_idx, n_idx], x_target_set[task_idx, n_idx],
                     y_support_set[task_idx, n_idx], y_target_set[task_idx, n_idx])
                pop[dataset_idx].append({'class': c, 'score': scores[task_idx, n_idx], 'label': true_labels[task_idx][n_idx]})

        if pool is self.pool:   # apply kmeans for new class instances and instances in the pool
            for column_idx, column in enumerate(pool['pool']):
                dataset_idx = int(column_idx / num_columns_per_dataset)
                for indi in column:
                    pop[dataset_idx].append({'class': indi['class'], 'score': indi['score'], 'label': indi['label']})
            pool['pool'] = [[] for _ in range(dim)]

            # apply kmeans separately for each dataset
            for dataset_idx in dataset_idxs_set:
                if len(pop[dataset_idx]) < dim:
                    # if no instance for 1 dataset, just continue
                    # if the instance number is smaller than num_clusters(dim), just continue,
                    # random put instances to pool according to centers
                    if dataset_idx == 0:
                        pool['centers'] = self.pool_rng.randn(num_columns_per_dataset, score_dim)
                    else:
                        pool['centers'] = np.concatenate([pool['centers'], self.pool_rng.randn(num_columns_per_dataset, score_dim)], axis=0)
                    continue

                # apply sklearn kmeans
                score_data = np.stack([item['score'] for item in pop[dataset_idx]])
                km = KMeans(n_clusters=num_columns_per_dataset, max_iter=50, init='k-means++', random_state=self.pool_rng)
                column_idxs = km.fit_predict(score_data)     # [len(pop)]
                centroids = km.cluster_centers_
                if dataset_idx == 0:
                    pool['centers'] = centroids
                else:
                    pool['centers'] = np.concatenate([pool['centers'], centroids], axis=0)

                # # ------debug-----------
                # print('centroids: \n', centroids)
                # # ------debug-----------

                # put each class instance into the pool
                for pop_idx, indi in enumerate(pop[dataset_idx]):
                    column_idx = column_idxs[pop_idx] + dataset_idx * num_columns_per_dataset   # to the correct column
                    dist = np.sqrt(np.sum(np.square(score_data[pop_idx] - centroids[column_idxs[pop_idx]])))

                    # put indi to column_idx's column
                    if not reduce:      # just put into the pool
                        pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                         'label': indi['label'],
                                                         'dist': dist})
                    else:
                        same_label_idxs = []
                        for idx, item in enumerate(pool['pool'][column_idx]):
                            if item['label'] == indi['label']:
                                same_label_idxs.append(idx)
                        dist_list = np.array([item['dist'] for item in pool['pool'][column_idx]])

                        # if pool['pool'][column_idx] contain this label, just compare this 2 c's dist
                        if len(same_label_idxs) > 0 and dist_list[same_label_idxs][0] > dist:       # should be only 1 since put 1 by 1.
                            pool['pool'][column_idx].pop(same_label_idxs[0])
                            pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                             'label': indi['label'],
                                                             'dist': dist})
                        # else just put.
                        elif len(same_label_idxs) == 0:
                            pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                             'label': indi['label'],
                                                             'dist': dist})

                            # delete the largest dist one to maintain pool_max_size items.
                            if len(pool['pool'][column_idx]) == self.pool_max_size + 1:
                                pool['pool'][column_idx].pop(
                                    int(np.argmax(np.array([item['dist'] for item in pool['pool'][column_idx]]))))

        elif pool is self.val_pool:     # just put new class instances into val_pool according to the trained centers

            for dataset_idx in dataset_idxs_set:
                centers = self.pool['centers'][dataset_idx * num_columns_per_dataset:
                                               (dataset_idx + 1) * num_columns_per_dataset]

                for pop_idx, indi in enumerate(pop[dataset_idx]):
                    dists = np.sqrt(np.sum((centers - indi['score']) ** 2, axis=1))
                    dists_idx = np.argmin(dists)
                    column_idx = dists_idx + dataset_idx * num_columns_per_dataset   # to the correct column

                    # put indi to column_idx's column
                    if not reduce:      # just put into the pool
                        pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                         'label': indi['label'],
                                                         'dist': dists[dists_idx]})
                    else:
                        same_label_idxs = []
                        for idx, item in enumerate(pool['pool'][column_idx]):
                            if item['label'] == indi['label']:
                                same_label_idxs.append(idx)
                        dist_list = np.array([item['dist'] for item in pool['pool'][column_idx]])

                        # if pool['pool'][column_idx] contain this label, just compare this 2 c's dist
                        if len(same_label_idxs) > 0 and dist_list[same_label_idxs][0] > dists[dists_idx]:
                            pool['pool'][column_idx].pop(same_label_idxs[0])
                            pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                             'label': indi['label'],
                                                             'dist': dists[dists_idx]})

                        # else just put.
                        elif len(same_label_idxs) == 0:
                            pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                             'label': indi['label'],
                                                             'dist': dists[dists_idx]})

                            # delete the largest dist one to maintain pool_max_size items.
                            if len(pool['pool'][column_idx]) == self.pool_max_size + 1:
                                pool['pool'][column_idx].pop(
                                    int(np.argmax(np.array([item['dist'] for item in pool['pool'][column_idx]]))))

        # sort each column in an ascending distance order.
        for cl_idx, column in enumerate(pool['pool']):
            dist_list = np.array([item['dist'] for item in column])
            sorted_idxs = np.argsort(dist_list)
            pool['pool'][cl_idx] = [column[idx] for idx in sorted_idxs]

    @torch.no_grad()
    def pool_assign(self, pop, pool=None, reduce=False):
        """
        pool assignment: for each class instance, assign this into the pool according to the closest cluster_center.
        Args:
            pop:
            pool:
            reduce: False to not reduce the number and consider true labels; true to do so.

        Returns:

        """
        if pool is None:
            pool = self.pool

        assert 'pool' in pool.keys() and 'centers' in pool.keys()

        cluster_layer_0 = self.args.cluster_layer_0
        cluster_layer_1 = self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1
        dim = self.pool_size

        # init pool
        pool['pool'] = [[] for _ in range(dim)]

        for indi in pop:
            dists = np.sqrt(np.sum((pool['centers'] - indi['score']) ** 2, axis=1))
            column_idx = np.argmin(dists)

            # put indi to column_idx's column
            if not reduce:      # just put into the pool
                pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'], 'label': indi['label'],
                                                 'dist': dists[column_idx]})

            else:
                same_label_idxs = []
                for idx, item in enumerate(pool['pool'][column_idx]):
                    if item['label'] == indi['label']:
                        same_label_idxs.append(idx)
                dist_list = np.array([item['dist'] for item in pool['pool'][column_idx]])

                # if pool['pool'][column_idx] contain this label, just compare this 2 c's dist
                if len(same_label_idxs) > 0 and dist_list[same_label_idxs][0] > dists[column_idx]:
                    pool['pool'][column_idx].pop(same_label_idxs[0])
                    pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                     'label': indi['label'],
                                                     'dist': dists[column_idx]})

                # else just put.
                elif len(same_label_idxs) == 0:
                    pool['pool'][column_idx].append({'class': indi['class'], 'score': indi['score'],
                                                     'label': indi['label'],
                                                     'dist': dists[column_idx]})

                    # delete the largest dist one to maintain pool_max_size items.
                    if len(pool['pool'][column_idx]) == self.pool_max_size + 1:
                        pool['pool'][column_idx].pop(
                            int(np.argmax(np.array([item['dist'] for item in pool['pool'][column_idx]]))))

    @torch.no_grad()
    def pool_center_update(self, pool=None):
        """
        already has pool['pool'] filled with the closest distance to the previous cluster centers.
        update cluster centers in pool['centers'] using the class instances' scores stored in the pool as the average.
        Args:
            pool:

        Returns:

        """
        if pool is None:
            pool = self.pool

        assert 'pool' in pool.keys() and 'centers' in pool.keys()
        assert type(pool['centers']) is np.ndarray
        for idx in range(len(pool['pool'])):
            assert len(pool['pool'][idx]) > 0

        for column_idx, column in enumerate(pool['pool']):
            pool['centers'][column_idx] = np.mean(np.stack(
                [item['score'] for item in pool['pool'][column_idx]]
            ), axis=0)

    @torch.no_grad()
    def pool_put(self, task_batch, true_labels, pool=None, reduce=True, dataset_idxs=None, flag='during'):
        """
        pool_put 是先求各个task_batch中bs*N个class的score （class_emb counts support 和 query set中的每一张图片）
        然后调用pool_put_all把class instances按照 class的score来放到pool中。
        Args:
            task_batch: batch of tasks
            true_labels: [batch_size, N]
            pool: the target pool
            reduce: False to not reduce the number and consider true labels; true to do so.
        Returns:  some information for debug
            assigns: [bs, N, cluster_layer_0]
            gates: [bs, N, cluster_layer_0, cluster_layer_1]
            cluster_idxs: [bs, N]
            scores: [bs, N, cluster_layer_0 * cluster_layer_1]
        """
        if pool is None:
            pool = self.pool

        x_support_set, x_target_set, y_support_set, y_target_set = task_batch
        # [batch_size, n_way, shot/Kq, c, h, w]; [batch_size, n_way, shot/Kq]
        # class: [shot/Kq, c, h, w]; [shot/Kq]
        if self.is_regression:
            bs, n, s, input_dim = x_support_set.shape
            _, _, kq, input_dim = x_target_set.shape
        else:
            bs, n, s, c, h, w = x_support_set.shape
            _, _, kq, _, _, _ = x_target_set.shape

        cluster_layer_0 = self.args.cluster_layer_0
        cluster_layer_1 = self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1

        # print("--------------debug---------------")
        # print('pool put: true_labels')
        # print(true_labels)

        assert len(true_labels) == bs
        assert len(true_labels[0]) == n
        assert not x_support_set.is_cuda

        assigns, gates, cluster_idxs, scores = [], [], [], []
        # for each class instance   bs*N
        for task_idx in range(bs):
            for n_idx in range(n):
                class_batch = (x_support_set[task_idx, n_idx].unsqueeze(dim=0),
                               x_target_set[task_idx, n_idx].unsqueeze(dim=0),
                               y_support_set[task_idx, n_idx].unsqueeze(dim=0),
                               y_target_set[task_idx, n_idx].unsqueeze(dim=0))
                # [1, s/kq, c, h, w]
                _, class_emb_vec = self.class_batch_2_class_embs(class_batch, n)   # [1, 1, 128]
                class_emb_vec = class_emb_vec[0]  # [1, 128]

                assert class_emb_vec.shape[0] == 1

                assign, gate, cluster_idx, score = \
                    self.taskSpecificNet.forward_task_emb_get_clustering_information(class_emb_vec)
                assigns.append(assign)
                gates.append(gate)
                cluster_idxs.append(cluster_idx)
                scores.append(score)

        assigns = np.concatenate(assigns)               # [bs*n, 4]
        gates = np.concatenate(gates)                   # [bs*n, 4, 2]
        cluster_idxs = np.concatenate(cluster_idxs)     # [bs*n]
        scores = np.concatenate(scores)                 # [bs*n, 8]

        assigns = assigns.reshape((bs, n, cluster_layer_0))
        gates = gates.reshape((bs, n, cluster_layer_0, cluster_layer_1))
        cluster_idxs = cluster_idxs.reshape((bs, n))
        scores = scores.reshape((bs, n, cluster_layer_0 * cluster_layer_1))

        # self.pool_put_kmeans_cheat(task_batch, true_labels, scores,
        #                            dataset_idxs=dataset_idxs, pool=pool, reduce=reduce, flag=flag)
        self.pool_put_kmeans(task_batch, true_labels, scores, pool=pool, reduce=reduce, flag=flag)

        return assigns, gates, cluster_idxs, scores

    def pop_class(self, cluster_idx, pool=None):
        """
        check C in pool if len(C) excess pool_max_size
        and pop one from C
        After that, len(C) is pool_max_size-1 and allow one c to be append in
        Args:
            pool: the target pool
            cluster_idx:

        Returns:

        """
        if pool is None:
            pool = self.pool

        while len(pool[cluster_idx]) > self.pool_max_size:    # delete the smallest one to maintain pool_max_size items.
            pool[cluster_idx].pop(int(np.argmin(np.array([item['assign'] for item in pool[cluster_idx]]))))

    def check_pool(self, pool=None, max=10):
        ## larger than desire num
        # 就return false
        if pool is None:
            pool = self.pool

        cluster_layer_0 = self.args.cluster_layer_0
        cluster_layer_1 = self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1

        # num_rich_queue = 0
        # for que in pool:
        #     if len(que) >= self.args.num_classes_per_set:
        #         num_rich_queue += 1
        # if num_rich_queue >= 4:
        #     return True
        # else:
        #     return False

        num_rich_queue = 0
        if type(pool) is dict:
            pool = pool['pool']
        for que in pool:
            if len(que) >= max:
                num_rich_queue += 1
        if num_rich_queue >= 2:
            return True
        else:
            return False

    def pool_get(self, num_obj=2, num_pop=2, num_mix=2, pool=None, for_augment=False):
        """
        assert num_pop >= num_obj
        choose num_pop columns (each column each task) as individuals
        from above volumns, the first num_obj columns also generate num_obj tasks as objectives
        for each num_mix mixture case, choose 2 columns (each column each task) to generate 1 mix.

        the total number of tasks are: num_obj + num_pop + 2*num_mix

        for val pool, usually be num_obj=8, num_pop=8, num_mix=2
        Args
            :param num_obj: num of sampled clusters for objectives
            :param num_pop: num of sampled clusters for population, should >= than num_obj
            :param num_mix: num of generated mix-task(-representation)
            :param for_augment: true if reture only pop and mix with query set; false normal
        Returns:
            a task_batch  [[_c1_][_c2_][c1][c2][cm11][cm12][cm21][cm22]]
            only the first num_obj tasks have query set.
        """
        assert num_pop >= num_obj
        if pool is None:
            pool = self.pool

        x_support_set, x_target_set, y_support_set, y_target_set = [], [], [], []
        # [batch_size, n_way, shot/Kq, c, h, w]; [batch_size, n_way, shot/Kq]
        # class: [shot/Kq, c, h, w]; [shot/Kq]

        available_idx = []
        # for que_idx, que in enumerate(pool):
        #     if len(que) >= self.args.num_classes_per_set:
        #         available_idx.append(que_idx)

        for que_idx, que in enumerate(pool['pool']):
            if len(que) >= self.args.num_classes_per_set:       # 5 for obj tasks, 5 for pop tasks. available overlap
                available_idx.append(que_idx)

        if len(available_idx) < num_pop:
            raise Exception("not enough samples in the queues")

        # select num_pop queues
        available_idx = np.array(available_idx)
        selected_idx = self.pool_rng.choice(available_idx, size=num_pop, replace=False)

        # repeat the first num_obj idx to the end to have len=num_obj+num_pop
        obj_idx = selected_idx[:num_obj]

        selected_idx = np.concatenate([obj_idx, selected_idx])

        # select 2 columns for 1 mix
        for mix_idx in range(num_mix):
            # mix choose the first and the last columns as those for objective
            selected_idx = np.concatenate([selected_idx, obj_idx[0:1], obj_idx[-1:]])
            # self.pool_rng.choice(selected_idx[:num_obj], size=2, replace=False)

        # print("-----------debug----------")
        # print("selected_idx:", selected_idx)

        # for each queues(clusters) generate 1 task; total: num_obj + num_pop + 2*num_mix
        sel_obj_class_idxs = []                             # the selected cls idxs for objectives
        for idx, que_idx in enumerate(selected_idx):        # for each selected queue: que_idx

            ## for objective tasks:
            if idx < len(obj_idx):
                # select and store the chosen class idxs for obj
                sel_obj_class_idxs.append(self.pool_rng.choice(
                    np.arange(len(pool['pool'][que_idx])), size=self.args.num_classes_per_set, replace=False))

                selected_class_idxs = sel_obj_class_idxs[idx]
            else:   ## for population tasks:
                # select num_classes_per_set classes from qsize() classes to get 1 task,
                # except cls in sel_obj_class_idxs
                candidate_class_idxs_list = np.arange(len(pool['pool'][que_idx]))

                # # if que_idx belongs to obj_idx,
                # # then remove the corresponding cls idxs in sel_obj_class_idxs
                # # the selected que should contain more than 10 classes.
                # if que_idx in obj_idx:
                #     sel_idx = list(obj_idx).index(que_idx)
                #     cls_idxs_need_remove = sel_obj_class_idxs[sel_idx]
                #     candidate_class_idxs_list = np.delete(candidate_class_idxs_list, cls_idxs_need_remove)

                selected_class_idxs = self.pool_rng.choice(
                    candidate_class_idxs_list, size=self.args.num_classes_per_set, replace=False)
                # 2-way maybe size = 2

            support_set_images, target_set_images = [], []
            if self.is_regression:
                support_set_labels, target_set_labels = [], []
            else:
                support_set_labels, target_set_labels = self.y_support_set, self.y_target_set

            for c_idx, selected_class_idx in enumerate(selected_class_idxs):
                c = pool['pool'][que_idx][selected_class_idx]['class']
                support_set_images.append(c[0])     # [shot, c, h, w]
                target_set_images.append(c[1])      # [Kq,   c, h, w]
                if self.is_regression:
                    support_set_labels.append(c[2])
                    target_set_labels.append(c[3])
            support_set_images = torch.stack(support_set_images)    # [n_way, shot, c, h, w]
            target_set_images = torch.stack(target_set_images)      # [n_way, Kq, c, h, w]
            if self.is_regression:
                support_set_labels = torch.stack(support_set_labels)
                target_set_labels = torch.stack(target_set_labels)

            x_support_set.append(support_set_images)
            y_support_set.append(support_set_labels)
            if for_augment:     # all task needs query set
                x_target_set.append(target_set_images)
                y_target_set.append(target_set_labels)
            elif idx < num_obj:   # only the first num_obj tasks need target set for query loss calculation
                x_target_set.append(target_set_images)
                y_target_set.append(target_set_labels)

        x_support_set = torch.stack(x_support_set)      # [8, n_way, shot, c, h, w]
        x_target_set = torch.stack(x_target_set)        # [2/8, n_way, Kq, c, h, w]
        y_support_set = np.stack(y_support_set)         # [8, n_way, shot]
        y_target_set = np.stack(y_target_set)           # [2/8, n_way, Kq]

        if for_augment:
            task_batch = (x_support_set[num_obj:], x_target_set[num_obj:],
                          y_support_set[num_obj:], y_target_set[num_obj:])
        else:
            task_batch = (x_support_set, x_target_set, y_support_set, y_target_set)
        return task_batch

    def get_cluster_centers(self):
        """
        get the cluster centers in hierarchical_clustering.assign_net
        Returns: cluster_centers: [4, 128]
        """
        return self.taskSpecificNet.hierarchicalClustering.assign_net.cluster_centers.data

    @torch.no_grad()
    def pool_reobtain_score(self):
        """
        every pool_update_rate epoch,
        for each class instance, update its score by re-following clustering net and cal center_update.
        Returns:  some information for debug
            assigns: [1, <=80, cluster_layer_0]
            gates: [1, <=80, cluster_layer_0, cluster_layer_1]
            cluster_idxs: [1, <=80]
            scores: [1, <=80, cluster_layer_0 * cluster_layer_1]
        """
        pool = self.pool
        assert 'pool' in pool.keys() and 'centers' in pool.keys()

        # re obtain clustering information for classes [<=8*10]
        assigns, gates, cluster_idxs, scores = [], [], [], []

        # for each class instance, update its score
        for column in pool['pool']:
            for indi in column:
                _, class_emb_vec = self.class_batch_2_class_embs(
                    (indi['class'][0].unsqueeze(0), indi['class'][1].unsqueeze(0),
                     indi['class'][2].unsqueeze(0), indi['class'][3].unsqueeze(0)),
                    self.args.num_classes_per_set)   # [1, 1, 128]
                class_emb_vec = class_emb_vec[0]    # [1, 128]
                assign, gate, cluster_idx, score = \
                    self.taskSpecificNet.forward_task_emb_get_clustering_information(class_emb_vec)
                assigns.append(assign)
                gates.append(gate)
                cluster_idxs.append(cluster_idx)
                scores.append(score)

                indi['score'] = score[0]        # [8]

        if pool['centers'] is not None:     # None at the very beginning, no class instances in the pool
            self.pool_center_update(pool)

        if len(assigns) > 0:
            assigns = np.concatenate(assigns)[np.newaxis, :]               # [1, 8*10, 4]
            gates = np.concatenate(gates)[np.newaxis, :]                   # [1, 8*10, 4, 2]
            cluster_idxs = np.concatenate(cluster_idxs)[np.newaxis, :]     # [1, 8*10]
            scores = np.concatenate(scores)[np.newaxis, :]                 # [1, 8*10, 8]

        return assigns, gates, cluster_idxs, scores

    def mixup_task_representation(self, data_batch, num_mix=2, val=False):
        ## get theta_i for mix_task_representations,
        ## each from 2 randomly sampled cluster tasks: data_batch_for_mix
        adapted_weights_batch_mix = []

        x_support_set, y_support_set = data_batch

        assert num_mix == 2
        assert x_support_set.shape[0] == 4
        for mix_id in range(num_mix):       # 2 mix task representations
            x_support_set_t, y_support_set_t = \
                x_support_set[2*mix_id:2*(mix_id+1)], y_support_set[2*mix_id:2*(mix_id+1)]
            if val:
                with torch.no_grad():
                    task_embs = self.taskSpecificNet.forward_get_task_emb(
                        x_support_set_t, y_support_set_t)   # [2 * [N, K, 128]]
                    task_embs = torch.stack(task_embs)      # [2, N, K, 128]
                    task_embs = task_embs.view(task_embs.shape[0], -1, task_embs.shape[3])    # [2, N*K, 128]
                    task_embs = torch.mean(task_embs, dim=1)       # [2, 128]
            else:
                task_embs = self.taskSpecificNet.forward_get_task_emb(
                    x_support_set_t, y_support_set_t)   # [2 * [N, K, 128]]
                task_embs = torch.stack(task_embs)      # [2, N, K, 128]
                task_embs = task_embs.view(task_embs.shape[0], -1, task_embs.shape[3])    # [2, N*K, 128]
                task_embs = torch.mean(task_embs, dim=1)       # [2, 128]
            # mix 2 task_emb
            # if mix_id == 0:
            #     lam = self.pool_rng.beta(a=5, b=2, size=1)[0]
            # else:
            #     lam = self.pool_rng.beta(a=2, b=5, size=1)[0]
            # lam = self.pool_rng.beta(a=self.args.mixup_alpha, b=self.args.mixup_beta, size=1)[0]
            if mix_id == 0:
                lam = torch.from_numpy(self.pool_rng.beta(a=5, b=2, size=task_embs.shape[-1])).double().to(self.device)     # 128
            else:
                lam = torch.from_numpy(self.pool_rng.beta(a=2, b=5, size=task_embs.shape[-1])).double().to(self.device)     # 128

            mix_task_embs = (lam*task_embs[0]+(1-lam)*task_embs[1]).unsqueeze(0)    # [1, 128]

            if val:
                with torch.no_grad():
                    adapted_weights_batch_mix.extend(self.taskSpecificNet.forward_task_emb(mix_task_embs))

                for adapted_weights in adapted_weights_batch_mix:
                    for key, item in adapted_weights.items():
                        item.requires_grad = True
            else:
                adapted_weights_batch_mix.extend(self.taskSpecificNet.forward_task_emb(mix_task_embs))

        return adapted_weights_batch_mix

    def mixup_imgs(self, data_batch, num_mix=2, val=False):
        ## get theta_i for mix_tasks

        x_support_set, y_support_set = data_batch

        assert num_mix == 2
        assert x_support_set.shape[0] == 4

        # [batch_size, n_way, shot, c, h, w]; [batch_size, n_way, shot]
        if self.is_regression:
            batch_size, n, shot, input_dim = x_support_set.shape
        else:
            batch_size, n, shot, c, h, w = x_support_set.shape

        ## obtain mix images
        x_support_set_mix = []
        y_support_set_mix = []
        # [0] and [int(batch_size/2)]
        for mix_id in range(num_mix):
            # generate lam for each images
            if mix_id == 0:
                lam = torch.from_numpy(
                    self.pool_rng.beta(
                        a=5, b=2,
                        size=(n, shot))
                ).double().to(self.device)     # [5,1]
            else:
                lam = torch.from_numpy(
                    self.pool_rng.beta(
                        a=2, b=5,
                        size=(n, shot))
                ).double().to(self.device)

            # for each image
            x_support_set_t, y_support_set_t = \
                x_support_set[2*mix_id:2*(mix_id+1)], y_support_set[2*mix_id:2*(mix_id+1)]
            # [2, n, shot, c, h, w]

            mix_imgs = []
            mix_labs = []
            for n_idx in range(n):
                for s_idx in range(shot):
                    img1, img2 = x_support_set_t[0, n_idx, s_idx].clone().detach(), \
                                 x_support_set_t[1, n_idx, s_idx].clone().detach()
                    lab1, lab2 = y_support_set_t[0, n_idx, s_idx].clone().detach(), \
                                 y_support_set_t[1, n_idx, s_idx].clone().detach()
                    mix_img = lam[n_idx, s_idx] * img1 + (1-lam[n_idx, s_idx]) * img2
                    mix_lab = lam[n_idx, s_idx] * lab1 + (1-lam[n_idx, s_idx]) * lab2
                    mix_imgs.append(mix_img)
                    mix_labs.append(mix_lab)
            if self.is_regression:
                mix_imgs = torch.stack(mix_imgs).reshape(n, shot, input_dim)
            else:
                mix_imgs = torch.stack(mix_imgs).reshape(n, shot, c, h, w)
            mix_labs = torch.stack(mix_labs).reshape(n, shot)
            x_support_set_mix.append(mix_imgs)
            y_support_set_mix.append(mix_labs)
        x_support_set_mix = torch.stack(x_support_set_mix)      # [2, n, shot, c, h, w]
        y_support_set_mix = torch.stack(y_support_set_mix)
        if not self.is_regression:              # classification
            y_support_set_mix = y_support_set_t

        ## obtain adapted_weights
        if val:
            with torch.no_grad():
                _, adapted_weights_batch_mix, _ = self.taskSpecificNet(x_support_set_mix, y_support_set_mix,
                                                                       detach_theta0=True)
            for adapted_weights in adapted_weights_batch_mix:
                for key, item in adapted_weights.items():
                    item.requires_grad = True
        else:
            _, adapted_weights_batch_mix, _ = self.taskSpecificNet(x_support_set_mix, y_support_set_mix,
                                                                   detach_theta0=True)

        return adapted_weights_batch_mix

    def segmix_reg(self, data_batch, num_mix=2, val=False):
        ## get theta_i for mix_tasks  segment mix for regression.
        ## TBD

        x_support_set, y_support_set = data_batch

        assert num_mix == 2
        assert x_support_set.shape[0] == 4

        # [batch_size, n_way, shot, c, h, w]; [batch_size, n_way, shot]
        if self.is_regression:
            batch_size, n, shot, input_dim = x_support_set.shape
        else:
            batch_size, n, shot, c, h, w = x_support_set.shape

        ## obtain mix images
        x_support_set_mix = []
        y_support_set_mix = []
        # [0] and [int(batch_size/2)]
        for mix_id in range(num_mix):
            # generate lam for each images as the crossover position: > lam img1, < lam img2?
            if mix_id == 0:
                lam = torch.from_numpy(
                    self.pool_rng.beta(
                        a=5, b=2,
                        size=(n, shot))
                ).double().to(self.device)     # [5,1]
            else:
                lam = torch.from_numpy(
                    self.pool_rng.beta(
                        a=2, b=5,
                        size=(n, shot))
                ).double().to(self.device)

            # for each image
            x_support_set_t, y_support_set_t = \
                x_support_set[2*mix_id:2*(mix_id+1)], y_support_set[2*mix_id:2*(mix_id+1)]
            # [2, n, shot, c, h, w]

            mix_imgs = []
            mix_labs = []
            for n_idx in range(n):
                for s_idx in range(shot):
                    img1, img2 = x_support_set_t[0, n_idx, s_idx].clone().detach(), \
                                 x_support_set_t[1, n_idx, s_idx].clone().detach()
                    lab1, lab2 = y_support_set_t[0, n_idx, s_idx].clone().detach(), \
                                 y_support_set_t[1, n_idx, s_idx].clone().detach()
                    mix_img = lam[n_idx, s_idx] * img1 + (1-lam[n_idx, s_idx]) * img2
                    mix_lab = lam[n_idx, s_idx] * lab1 + (1-lam[n_idx, s_idx]) * lab2
                    mix_imgs.append(mix_img)
                    mix_labs.append(mix_lab)
            if self.is_regression:
                mix_imgs = torch.stack(mix_imgs).reshape(n, shot, input_dim)
            else:
                mix_imgs = torch.stack(mix_imgs).reshape(n, shot, c, h, w)
            mix_labs = torch.stack(mix_labs).reshape(n, shot)
            x_support_set_mix.append(mix_imgs)
            y_support_set_mix.append(mix_labs)
        x_support_set_mix = torch.stack(x_support_set_mix)      # [2, n, shot, c, h, w]
        y_support_set_mix = torch.stack(y_support_set_mix)
        if not self.is_regression:              # classification
            y_support_set_mix = y_support_set_t

        ## obtain adapted_weights
        if val:
            with torch.no_grad():
                _, adapted_weights_batch_mix, _ = self.taskSpecificNet(x_support_set_mix, y_support_set_mix,
                                                                       detach_theta0=True)
            for adapted_weights in adapted_weights_batch_mix:
                for key, item in adapted_weights.items():
                    item.requires_grad = True
        else:
            _, adapted_weights_batch_mix, _ = self.taskSpecificNet(x_support_set_mix, y_support_set_mix,
                                                                   detach_theta0=True)

        return adapted_weights_batch_mix

    def cutmix_imgs(self, data_batch, num_mix=2, val=False):
        ## get theta_i for mix_tasks

        x_support_set, y_support_set = data_batch

        assert num_mix == 2
        assert x_support_set.shape[0] == 4

        # [batch_size, n_way, shot, c, h, w]; [batch_size, n_way, shot]
        batch_size, n, shot, c, h, w = x_support_set.shape

        ## obtain mix images
        rng = self.pool_rng
        cutmix_prop = 0.3

        x_support_set_mix = []
        for mix_id in range(num_mix):
            # generate 2 masks for 2 imgs with size [c, h, w]
            cuth, cutw = int(h * cutmix_prop), int(w * cutmix_prop)
            # mask1, mask2 = np.zeros((c,h,w)), np.zeros((c,h,w))
            posih1 = rng.randint(h - cuth)
            posiw1 = rng.randint(w - cutw)
            # mask1[:, posih1:posih1 + cuth, posiw1:posiw1 + cutw] = 1
            posih2 = rng.randint(h - cuth)
            posiw2 = rng.randint(w - cutw)
            # mask2[:, posih2:posih2 + cuth, posiw2:posiw2 + cutw] = 1
            # mask1 = torch.from_numpy(mask1).double().to(self.device)
            # mask2 = torch.from_numpy(mask2).double().to(self.device)

            # generate lam for each image, used to decide the background and foreground.
            if mix_id == 0:
                lam = torch.from_numpy(
                    self.pool_rng.beta(
                        a=5, b=2,
                        size=(n, shot))
                ).double().to(self.device)     # [5,1]
            else:
                lam = torch.from_numpy(
                    self.pool_rng.beta(
                        a=2, b=5,
                        size=(n, shot))
                ).double().to(self.device)


            # for each image
            x_support_set_t, y_support_set_t = \
                x_support_set[2*mix_id:2*(mix_id+1)], y_support_set[2*mix_id:2*(mix_id+1)]
            # [2, n, shot, c, h, w]

            #
            mix_imgs = []
            for n_idx in range(n):
                for s_idx in range(shot):
                    img1, img2 = x_support_set_t[0, n_idx, s_idx], x_support_set_t[1, n_idx, s_idx]

                    # if lam > 0.5, 用img1做background(0)，img2做cut部分(1)
                    if lam[n_idx, s_idx] >= 0.5:
                        mix_img = img1.clone().detach()
                        mix_img[:, posih1:posih1 + cuth, posiw1:posiw1 + cutw] = \
                            img2[:, posih2:posih2 + cuth, posiw2:posiw2 + cutw].clone().detach()
                    else:   # if lam < 0.5, 用img2做background(0)，img1做cut部分(1)
                        mix_img = img2.clone().detach()
                        mix_img[:, posih2:posih2 + cuth, posiw2:posiw2 + cutw] = \
                            img1[:, posih1:posih1 + cuth, posiw1:posiw1 + cutw].clone().detach()

                    mix_imgs.append(mix_img)
            mix_imgs = torch.stack(mix_imgs).reshape(n, shot, c, h, w)
            x_support_set_mix.append(mix_imgs)
        x_support_set_mix = torch.stack(x_support_set_mix)  # [2, n, shot, c, h, w]

        ## obtain adapted_weights
        if val:
            with torch.no_grad():
                _, adapted_weights_batch_mix, _ = self.taskSpecificNet(x_support_set_mix, y_support_set_t,
                                                                       detach_theta0=True)
            for adapted_weights in adapted_weights_batch_mix:
                for key, item in adapted_weights.items():
                    item.requires_grad = True
        else:
            _, adapted_weights_batch_mix, _ = self.taskSpecificNet(x_support_set_mix, y_support_set_t,
                                                                   detach_theta0=True)

        return adapted_weights_batch_mix

    def get_zero_conflict_losses(self, obj_size, pop_size):
        """
        for case that before pool_start_epoch or pool not available
        to align the conflict loss,
        we generate all zero parts for:
        loss,       {pop_idx}_loss,     {pop_idx}_loss_{obj_idx}
        accuracy,   {pop_idx}_accuracy, {pop_idx}_accuracy_{obj_idx}
        Returns: conflict_losses
        """
        conflict_losses = dict()

        ## the order of the assignment matters!! 完全按照与正常得到conflict losses一样的顺序来赋值
        # 0_loss
        # 0_loss_{0-3}
        # 0_accuracy
        # 0_accuracy_{0-3}
        # 1...
        # 2...
        # 3...
        # loss
        # accuracy

        for pop_idx in range(pop_size):
            conflict_losses['{}_loss'.format(pop_idx)] = 0
            for obj_idx in range(obj_size):
                conflict_losses['{}_loss_{}'.format(pop_idx, obj_idx)] = 0
            conflict_losses['{}_accuracy'.format(pop_idx)] = 0
            for obj_idx in range(obj_size):
                conflict_losses['{}_accuracy_{}'.format(pop_idx, obj_idx)] = 0

        conflict_losses['loss'] = 0
        conflict_losses['accuracy'] = 0

        return conflict_losses

    def get_zero_losses(self, obj_size):
        losses = dict()

        losses['loss'] = torch.tensor(0.0).to(self.device)
        for idx in range(obj_size):
            losses['loss_{}'.format(idx)] = torch.tensor(0.0).to(self.device)
        losses['accuracy'] = 0.0
        for idx in range(obj_size):
            losses['accuracy_{}'.format(idx)] = 0.0
        return losses

    def get_pool_size(self, pool=None):
        """
        return a vector representing task_pool's qsize() for each queue
        Returns:

        """
        if pool is None:
            pool = self.pool

        pool_size = [len(pool[idx]) for idx in range(len(pool))]
        return np.array(pool_size)

    def get_pool_size_fig(self, pool=None):
        """
        return a fig of bar for pool_size vector
        Returns:

        """
        if pool is None:
            pool = self.pool

        fig = plt.figure()
        plt.bar(x=np.arange(len(pool)), height=self.get_pool_size(pool), width=1)
        plt.xlim([-0.5, len(pool)-0.5])
        plt.xlabel('Queue idxs')
        plt.ylabel('# tasks')

        return fig

    def build_prefix_dict(self, losses, prefix, prefixed_losses=None):
        """
        Build/Updates a prefixed_losses with the key: {prefix}_{key}
        Args:
            losses: dict before prefixed
            prefix: 'train' or 'conflict'
            prefixed_losses: the dict after prefixed

        Returns: prefixed_losses
        """
        if prefixed_losses is None:
            prefixed_losses = dict()

        for key in losses:
            prefixed_losses["{}_{}".format(prefix, key)] = losses[key]

        return prefixed_losses

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        if type(total_losses) is list:
            total_losses = torch.stack(total_losses)
        losses['loss'] = torch.mean(total_losses)
        for idx in range(len(total_losses)):
            losses['loss_{}'.format(idx)] = total_losses[idx]
        losses['accuracy'] = np.mean(total_accuracies).item()
        for idx in range(len(total_accuracies)):
            losses['accuracy_{}'.format(idx)] = total_accuracies[idx].item()
        return losses

    def forward(self, data_batch, epoch, training_phase, test_phase=False, detach_theta0=False, medium_record=False):
        """
        :param data_batch: A enriched data batch containing the support and target sets.
            :x_support_set: support_x batch: [meta_batch_size, N, Ks, 3, img_size, img_size]
            :y_support_set: support_y batch: [meta_batch_size, N, Ks]
            :x_target_set:  query_x   batch: [meta_batch_size, N, Kq, 3, img_size, img_size]
            :y_target_set:  query_y   batch: [meta_batch_size, N, Kq]
        :param epoch: Current epoch's index
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :param test_phase: True -> test_phase, default False
        :return:
            :losses: A dictionary with the collected losses of the current outer forward propagation.
            :adapted_weights_batch: list of adapted weights (theta_i) for the batch
            :medium_batch: medium vecs for debug visualization.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch
        batch_size = x_support_set.shape[0]
        x, y = self.batch_reshape_back(data_batch)

        if test_phase or not training_phase:          # test or val
            # train task embedding using only support set
            with torch.no_grad():
                task_emb_loss_batch, adapted_weights_batch, medium_batch = self.taskSpecificNet(
                    x_support_set, y_support_set, medium_record=medium_record)

            for adapted_weights in adapted_weights_batch:
                for key, item in adapted_weights.items():
                    item.requires_grad = True

            # losses, per_task_target_preds = self.taskSpecificNet.maml_module.evaluation_forward_val_prop(
            #     data_batch, epoch, adapted_weights_batch)
            loss, y_pred = meta_gradient_step(
                self.taskSpecificNet.maml_module, self.optimizer, self.loss_fn, x=x, y=y, epoch=epoch,
                n_shot=self.args.num_samples_per_class, k_way=self.args.num_classes_per_set,
                q_queries=self.args.num_target_samples,
                order=self.args.order, inner_lr=self.args.inner_learning_rate,
                device=self.device, backward=False,
                train=False, inner_train_steps=self.args.inner_val_steps,
                adapted_weights_batch=adapted_weights_batch)
            # loss [bs,]; y_pred [bs, s-way*q_query, s-way]
            assert y_pred.shape == (batch_size, self.args.num_classes_per_set*self.args.num_target_samples,
                                    self.args.num_classes_per_set)

            acc = torch.eq(
                y_pred.argmax(dim=-1),
                self.q_relative_labels.repeat(batch_size).reshape(batch_size, -1)
            ).sum(dim=1).detach().cpu().numpy() / y_pred.shape[1]    # acc [bs, ]
            losses = self.get_across_task_loss_metrics(total_losses=loss, total_accuracies=acc)
            per_task_target_preds = y_pred

        else:
            # train task embedding using query set
            # task_emb_loss_batch, adapted_weights_batch, medium_batch = self.taskSpecificNet(
            #     torch.cat([x_support_set, x_target_set], dim=2),
            #     torch.cat([y_support_set, y_target_set], dim=2))

            # train task embedding using only support set
            task_emb_loss_batch, adapted_weights_batch, medium_batch = self.taskSpecificNet(
                x_support_set, y_support_set, detach_theta0=detach_theta0, medium_record=medium_record)

            # losses, per_task_target_preds = self.taskSpecificNet.maml_module.train_forward_val_prop(
            #     data_batch, epoch, adapted_weights_batch)
            loss, y_pred = meta_gradient_step(
                self.taskSpecificNet.maml_module, self.optimizer, self.loss_fn, x=x, y=y, epoch=epoch,
                n_shot=self.args.num_samples_per_class, k_way=self.args.num_classes_per_set,
                q_queries=self.args.num_target_samples,
                order=self.args.order, inner_lr=self.args.inner_learning_rate,
                device=self.device, backward=False,
                train=True, inner_train_steps=self.args.inner_train_steps,
                adapted_weights_batch=adapted_weights_batch)
            # loss [bs,]; y_pred [bs, s-way*q_query, s-way]
            assert y_pred.shape == (batch_size, self.args.num_classes_per_set*self.args.num_target_samples,
                                    self.args.num_classes_per_set)

            acc = torch.eq(
                y_pred.argmax(dim=-1),
                self.q_relative_labels.repeat(batch_size).reshape(batch_size, -1)
            ).sum(dim=1).detach().cpu().numpy() / y_pred.shape[1]    # acc [bs, ]
            losses = self.get_across_task_loss_metrics(total_losses=loss, total_accuracies=acc)
            per_task_target_preds = y_pred

        if medium_record:
            for aw_idx, adapted_weights in enumerate(adapted_weights_batch):
                ## adapted_weights_batch to numpy
                adapted_weights_batch[aw_idx] = {
                    key: item.detach().cpu().numpy() for (key, item) in adapted_weights.items()
                }
        else:
            adapted_weights_batch = []

        losses['recon_loss'] = torch.mean(task_emb_loss_batch)
        for idx in range(len(task_emb_loss_batch)):
            losses['recon_loss_{}'.format(idx)] = task_emb_loss_batch[idx]

        return losses, per_task_target_preds, adapted_weights_batch, medium_batch

    def pool_forward(self, data_batch, epoch, num_obj=2, num_pop=2, num_mix=2, val=False, medium_record=False):
        """
        pool forward: calculate the loss/acc of data_batch[2:],
        on data_batch[:2].
        :param data_batch: A enriched data batch containing the support and target sets.
            [[_c1_][_c2_][c3][c4][c1][c2][mix]...]
            :x_support_set: support_x batch: [2+2+4, N, Ks, 3, img_size, img_size]
            :y_support_set: support_y batch: [2+2+4, N, Ks]
            :x_target_set:  query_x   batch: [2, N, Kq, 3, img_size, img_size]
            :y_target_set:  query_y   batch: [2, N, Kq]
        :param num_obj: num of tasks for objective, have target set
        :param num_pop: num of tasks for population, have no target set
        :param num_mix: num of mixed task representation generated, have no target set. 2 means 4 tasks (2 each)
        :param epoch: Current epoch's index
        :return:
            :losses: A dictionary with the collected losses of the current outer forward propagation.
            :adapted_weights_batch: list of numpy-ed adapted weights (theta_i) for theta_is of cluster_centers and mix tasks
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        data_batch_for_loss = (x_support_set[:num_obj],
                               x_target_set[:num_obj],
                               y_support_set[:num_obj],
                               y_target_set[:num_obj])
        data_batch_for_pop = (x_support_set[num_obj:num_obj+num_pop],
                              y_support_set[num_obj:num_obj+num_pop])

        data_batch_for_mix = (x_support_set[num_obj+num_pop:],
                              y_support_set[num_obj+num_pop:])

        ## for all theta_i, obtain loss matrix
        # val adapted_weights_batch on data_batch_for_loss, to have a obj matrix
        losses_list = []
        adapted_weights_batch = []
        adapted_weights_batch_np = []

        ## get theta_i for all data_batch_for_pop
        x_support_set, y_support_set = data_batch_for_pop

        if val:
            with torch.no_grad():
                _, adapted_weights_batch_pop, _ = self.taskSpecificNet(x_support_set, y_support_set, detach_theta0=True)
            for adapted_weights in adapted_weights_batch_pop:
                for key, item in adapted_weights.items():
                    item.requires_grad = True
        else:
            _, adapted_weights_batch_pop, _ = self.taskSpecificNet(x_support_set, y_support_set, detach_theta0=True)

        adapted_weights_batch.extend(adapted_weights_batch_pop)

        ## get theta_i for all data_batch_for_mix
        if hasattr(self.args, 'mix_method') and self.args.mix_method == 'mixup':
            adapted_weights_batch_mix = self.mixup_imgs(data_batch_for_mix, num_mix=num_mix, val=val)
        elif hasattr(self.args, 'mix_method') and self.args.mix_method == 'cutmix':
            adapted_weights_batch_mix = self.cutmix_imgs(data_batch_for_mix, num_mix=num_mix, val=val)
        else:
            # mixup task_representation
            adapted_weights_batch_mix = self.mixup_task_representation(data_batch_for_mix, num_mix=num_mix, val=val)

        adapted_weights_batch.extend(adapted_weights_batch_mix)

        # calculate conflict losses
        x, y = self.batch_reshape_back(data_batch_for_loss)

        for adapted_weights in adapted_weights_batch:
            if val:
                loss, y_pred = meta_gradient_step(
                    self.taskSpecificNet.maml_module, self.optimizer, self.loss_fn, x=x, y=y, epoch=epoch,
                    n_shot=self.args.num_samples_per_class, k_way=self.args.num_classes_per_set,
                    q_queries=self.args.num_target_samples,
                    order=self.args.order, inner_lr=self.args.inner_learning_rate,
                    device=self.device, backward=False,
                    train=False, inner_train_steps=self.args.inner_val_steps,
                    adapted_weights_batch=[adapted_weights for _ in range(num_obj)])
                # loss [bs,]; y_pred [bs, s-way*q_query, s-way]

            else:   # train
                loss, y_pred = meta_gradient_step(
                    self.taskSpecificNet.maml_module, self.optimizer, self.loss_fn, x=x, y=y, epoch=epoch,
                    n_shot=self.args.num_samples_per_class, k_way=self.args.num_classes_per_set,
                    q_queries=self.args.num_target_samples,
                    order=self.args.order, inner_lr=self.args.inner_learning_rate,
                    device=self.device, backward=False,
                    train=True, inner_train_steps=self.args.inner_train_steps,
                    adapted_weights_batch=[adapted_weights for _ in range(num_obj)])
                # loss [bs,]; y_pred [bs, s-way*q_query, s-way]

            acc = torch.eq(
                y_pred.argmax(dim=-1),
                self.q_relative_labels.repeat(num_obj).reshape(num_obj, -1)
            ).sum(dim=1).detach().cpu().numpy() / y_pred.shape[1]       # acc [num_obj, ]
            indi_losses = self.get_across_task_loss_metrics(total_losses=loss, total_accuracies=acc)

            if val:
                # detach loss here
                for key, item in indi_losses.items():
                    if type(item) is torch.Tensor:
                        indi_losses[key] = item.detach().cpu()

            losses_list.append(indi_losses)

            if medium_record:
                adapted_weights_t = {key: item.detach().cpu().numpy() for (key, item) in adapted_weights.items()}
                adapted_weights_batch_np.append(adapted_weights_t)

        losses = self.get_accross_indi_loss_metrics(losses_list)  # after that, this losses['loss'] is np.

        return losses, adapted_weights_batch_np

    def cal_contrastive_loss(self, data_batch):         # TBD
        """
        for 1 task: (5 classes in same cluster)
            pos_emb: random 1 class's support and query image embs [16, 128]
            neg_emb: other 4 classes' support image embs [4, 128]
        Args:
            data_batch: (x_support_set, x_target_set, y_support_set, y_target_set)
                        [bs(4), n(5), s(1 or 15), img_shape]
        Returns:

        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        loss = []
        task_embs = self.taskSpecificNet.forward_get_task_emb(torch.cat([
            x_support_set, x_target_set], dim=2), torch.cat([y_support_set, y_target_set], dim=2))
        # [bs * [N, K, 128]]
        # for task_idx in range(x_support_set.shape[0]):

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if hasattr(self.args, 'clamp') and self.args.clamp > 0:
            for name, param in self.named_parameters():
                # print('{} {} {} {}'.format(name, param.shape, param.is_leaf, param.requires_grad))
                if param.requires_grad and param.grad is not None:
                    param.grad.data.clamp_(-self.args.clamp, self.args.clamp)
        self.optimizer.step()

    def batch_reshape(self, x, y):
        """
        :param x: [bs, n_shot * k_way + q_quer * k_way, data_shape]
        :param y: {'q_relative_labels': [bs, q_quer * k_way]'true_labels': [bs, n_shot * k_way + q_quer * k_way]}
        :return: x_support_set: [bs, k_way, n_shot, data_shape]
        :return: x_target_set,  [bs, k_way, q_quer, data_shape]
        :return: y_support_set, [bs, k_way, n_shot]
        :return: y_target_set,  [bs, k_way, q_quer]
        :return: true_labels,   [bs, k_way]
        """
        y = y['true_labels']
        data_shape = x.shape[2:]
        batch_size = x.shape[0]
        n_shot = self.args.num_samples_per_class
        k_way = self.args.num_classes_per_set
        q_quer = self.args.num_target_samples
        assert x.shape[1] == n_shot * k_way + q_quer * k_way
        x_support_set = x[:, :n_shot * k_way].reshape(batch_size, k_way, n_shot, *data_shape)
        x_target_set  = x[:, n_shot * k_way:].reshape(batch_size, k_way, q_quer, *data_shape)

        if self.args.is_regression:
            y_support_set = y[:, :n_shot * k_way].reshape(batch_size, k_way, n_shot)
            y_target_set  = y[:, n_shot * k_way:].reshape(batch_size, k_way, q_quer)
        else:
            y_support_set = create_nshot_task_label(k_way, n_shot).repeat(batch_size).reshape(
                batch_size, k_way, n_shot).to(self.device)
            y_target_set  = create_nshot_task_label(k_way, q_quer).repeat(batch_size).reshape(
                batch_size, k_way, q_quer).to(self.device)

        true_labels = y[:, :n_shot * k_way].reshape(batch_size, k_way, n_shot)
        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        return data_batch, true_labels[:, :, 0].to('cpu')

    def batch_reshape_back(self, data_batch):
        """
        reshape back to x and y in few-shot framework
        :param data_batch:
                x_support_set: [bs, k_way, n_shot, data_shape]
                x_target_set,  [bs, k_way, q_quer, data_shape]
                y_support_set, [bs, k_way, n_shot]
                y_target_set,  [bs, k_way, q_quer]
        :return: x: [bs, n_shot * k_way + q_quer * k_way, data_shape]
        :return: y: [bs, n_shot * k_way + q_quer * k_way]
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch
        batch_size = x_support_set.shape[0]
        k_way, n_shot = x_support_set.shape[1], x_support_set.shape[2]
        q_quer = x_target_set.shape[2]
        data_shape = x_support_set.shape[3:]
        assert k_way == self.args.num_classes_per_set

        x = torch.cat(
            [x_support_set.reshape(batch_size, n_shot * k_way, *data_shape),
             x_target_set.reshape(batch_size, q_quer * k_way, *data_shape)], dim=1
        ).reshape(batch_size, n_shot * k_way + q_quer * k_way, *data_shape)
        y = torch.cat(
            [y_support_set.reshape(batch_size, n_shot * k_way),
             y_target_set.reshape(batch_size, q_quer * k_way)], dim=1
        ).reshape(batch_size, n_shot * k_way + q_quer * k_way)

        return x, y

    def run_train_iter(self, model: nn.Module,
                       optimiser: optim.Optimizer,
                       loss_fn: Callable,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       epoch,
                       medium_record=False):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param model: self
        :param optimiser: self.optimizer
        :param loss_fn: CrossEntropyLoss
        :param x: [bs, n_shot * k_way + q_query * k_way, data_shape]
        :param y: [bs, n_shot * k_way + q_query * k_way]
        :param epoch: the index of the current epoch, starting from 1
        :return: The losses of the ran iteration.
        """
        if not self.training:
            self.train()

        epoch = int(epoch)-1
        if self.current_epoch != epoch:
            self.current_epoch = epoch

            ## re-cluster the tasks in the pool
            if epoch % self.args.pool_update_rate == 0:
                # self.recluster_pool()
                self.pool_reobtain_score()

        data_batch_cuda, true_labels = self.batch_reshape(x, y)

        ## calculate train loss as standard HSML
        train_losses, per_task_target_preds, adapted_weights_batch, medium_batch = \
            self.forward(data_batch=data_batch_cuda, epoch=epoch, training_phase=True, medium_record=medium_record)

        # train_losses: loss, loss_{}, accuracy_{}, recon_loss, recon_loss_{}
        train_loss = train_losses['loss'] + self.args.emb_loss_weight * train_losses['recon_loss']

        if medium_record:
            # task instance clustering info
            _, _, medium_batch['cluster_idxs'], medium_batch['scores'] = \
                self.taskSpecificNet.structurize_clustering_information(medium_batch['assigns'], medium_batch['gates'])

        # put the data_batch into the pool
        if hasattr(self.args, "use_pool") and self.args.use_pool:
            # and self.trained_iterations % 10 == 0
            # 10 for train
            data_batch_cpu = (data_batch_cuda[0].to('cpu'), data_batch_cuda[1].to('cpu'),
                              data_batch_cuda[2].to('cpu'), data_batch_cuda[3].to('cpu'))
            assigns, gates, cluster_idxs, scores = self.pool_put(
                data_batch_cpu, true_labels, pool=self.pool, reduce=True, dataset_idxs=None)

        pool_flag = self.check_pool(max=self.args.num_classes_per_set * 2)

        HV_loss = 0.0
        num_obj, num_pop, num_mix = 2, 2, 2
        assert num_mix == 2         # mix策略是用Beta(5,2)和Beta(2,5)
        # if available conflict losses calculation
        cal_conflict_matrix_flag = pool_flag and epoch >= self.args.pool_start_epoch

        if cal_conflict_matrix_flag:
            # and self.trained_iterations % 10 == 0
            for_augment = self.args.use_augment if hasattr(self.args, "use_augment") else False

            structured_sampled_data_batch = self.pool_get(num_obj=num_obj, num_pop=num_pop, num_mix=num_mix,
                                                          for_augment=for_augment)

            x_support_set, x_target_set, y_support_set, y_target_set = structured_sampled_data_batch

            x_support_set = x_support_set.double().to(device=self.device)
            x_target_set = x_target_set.double().to(device=self.device)
            if hasattr(self.args, "is_regression") and self.args.is_regression:
                y_support_set = torch.Tensor(y_support_set).double().to(device=self.device)
                y_target_set = torch.Tensor(y_target_set).double().to(device=self.device)
            else:
                y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
                y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

            data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)
            # x_support_set [8, n_way, shot, c, h, w]
            # x_target_set  [2, n_way, Kq, c, h, w]
            # y_support_set [8, n_way, shot]
            # y_target_set  [2, n_way, Kq]

            if for_augment:     # not use MO, use as task augmentation
                conflict_losses = self.get_zero_conflict_losses(
                    obj_size=num_obj,
                    pop_size=num_pop + num_mix)
                aug_losses, _, _, _ = \
                    self.forward(data_batch=data_batch, epoch=epoch, training_phase=True, detach_theta0=True)
                train_loss = train_loss + self.args.aug_weight * aug_losses['loss']
            else:
                # conflict losses: on loss and accuracy
                conflict_losses, _ = self.pool_forward(
                    data_batch=data_batch, epoch=epoch, num_obj=num_obj, num_pop=num_pop, num_mix=num_mix, val=False,
                    medium_record=medium_record)

                ## losses
                # train_losses:
                #   loss, loss_{0-3}, accuracy_{0-3}, recon_loss, recon_loss_{0-3}  -> train_loss
                # conflict_losses:
                #   {0-3}_loss_{0,1; 2,3} cluster_center{0-3} theta_i on {c1; c2} tasks;
                #   {4..}_loss_{0,1; 2,3} mix theta_i on {c1; c2} tasks;

                # obtain pop_objs for conflict loss
                pop_objs = []
                for pop_idx in range(num_pop + num_mix):
                    # 4
                    objs = []
                    for obj_idx in range(num_obj):
                        obj = torch.mean(torch.stack(
                            [conflict_losses['{}_loss_{}'.format(pop_idx, item_idx)] for item_idx in
                             range(obj_idx * self.args.num_sampled_tasks_for_each_objective,
                                   (obj_idx + 1) * self.args.num_sampled_tasks_for_each_objective)]))
                        objs.append(obj)
                    objs = torch.stack(objs)
                    pop_objs.append(objs)       # [pop_size, obj_size]
                pop_objs = torch.stack(pop_objs)   # [4, 2]

                HV_loss = cal_HV_loss(pop_objs) if self.args.HV_weight != 0.0 else 0

        else:  # put but get not available or before start_epoch, so just train with data_batch
            conflict_losses = self.get_zero_conflict_losses(
                obj_size=num_obj,
                pop_size=num_pop + num_mix)
            # 4 for cluster_centers, 2 for mix-tasks

        cal_conflict_loss_flag = self.args.use_conflict_loss and cal_conflict_matrix_flag
        # calculate HV loss
        if cal_conflict_loss_flag:
            loss = train_loss + self.args.HV_weight * HV_loss

        else:   # do not use conflict loss, just standard HSML
            loss = train_loss

        self.meta_update(loss=loss)

        self.trained_iterations += 1

        conflict_losses['HV_loss'] = HV_loss

        losses = self.build_prefix_dict(train_losses, 'train')
        losses = self.build_prefix_dict(conflict_losses, 'conflict', prefixed_losses=losses)
        losses['total_loss'] = loss
        ## losses
        # train_loss, train_loss_{batch_size}, train_recon_loss, train_recon_loss_{batch_size},
        # conflict_loss, conflict_{pop_size}_loss, conflict_{pop_size}_loss_{2*num_sampled_tasks_for_each_objective},
        # accuracy similar with loss
        # conflict_HV_loss
        # total_loss

        # change all these to scalar.

        losses['adapted_weights_batch'] = adapted_weights_batch
        losses['medium_batch'] = medium_batch

        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_test_iter(self, model: nn.Module,
                      optimiser: optim.Optimizer,
                      loss_fn: Callable,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      epoch,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      train: bool,
                      medium_record=False):
        """
        Runs test only obtain test losses.
        :param model: self
        :param optimiser: self.optimizer
        :param loss_fn: CrossEntropyLoss
        :param x: [bs, n_shot * k_way + q_query * k_way, data_shape]
        :param y: [bs, n_shot * k_way + q_query * k_way]
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        if self.training:
            self.eval()

        data_batch_cuda, true_labels = self.batch_reshape(x, y)

        ## calculate train loss as standard HSML
        losses, per_task_target_preds, adapted_weights_batch, medium_batch = \
            self.forward(data_batch=data_batch_cuda, epoch=epoch, training_phase=False, test_phase=True,
                         medium_record=medium_record)

        if medium_record:
            # task instance clustering info
            _, _, medium_batch['cluster_idxs'], medium_batch['scores'] = \
                self.taskSpecificNet.structurize_clustering_information(medium_batch['assigns'], medium_batch['gates'])
        else:
            medium_batch['cluster_idxs'], medium_batch['scores'] = [], []

        # #--------debug-------------
        # assigns, gates, cluster_idxs, scores = self.debug_collect_class_clustering_info(data_batch)
        #
        # return losses, per_task_per_indi_target_preds, {
        #     'adapted_weights_batch': adapted_weights_batch, 'medium_batch': medium_batch,
        #     'assigns': assigns, 'gates': gates, 'cluster_idxs': cluster_idxs, 'scores': scores}
        # #--------debug-------------

        losses['adapted_weights_batch'] = adapted_weights_batch
        losses['medium_batch'] = medium_batch

        return losses, per_task_target_preds

    # no use
    def run_validation_iter(self, data_batch, true_labels, dataset_idxs=None, flag='during'):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param true_labels: [batch_size, N]
        :return: The losses of the ran iteration.
        """
        if self.training:
            self.eval()

        if self.val_iter == 0:
            # reset val_pool
            self.val_pool = self.init_pool()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch      # [4]

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        if hasattr(self.args, "is_regression") and self.args.is_regression:
            y_support_set = torch.Tensor(y_support_set).float().to(device=self.device)
            y_target_set = torch.Tensor(y_target_set).float().to(device=self.device)
        else:
            y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
            y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch_cuda = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds, adapted_weights_batch, medium_batch = \
            self.forward(data_batch=data_batch_cuda, epoch=self.current_epoch, training_phase=False)

        # task instance clustering info
        _, _, medium_batch['cluster_idxs'], medium_batch['scores'] = \
            self.taskSpecificNet.structurize_clustering_information(medium_batch['assigns'], medium_batch['gates'])

        ## 每个batch都放到val_pool里， max_pool_size等setting都用和train一样的。
        # assigns, gates = self.predict_clusters_assign_gate_for_each_class(data_batch_cuda)  # [batch_size, N, 4]
        # self.pool_put_all(data_batch, true_labels, assigns, gates, self.val_pool)  # put the data_batch into the pool

        self.val_iter = self.val_iter + 1
        reduce = True
        # if flag is 'after' and self.val_iter % int(self.args.num_evaluation_tasks / self.batch_size) != 0:
        # if self.val_iter % int(self.args.num_evaluation_tasks / self.batch_size) != 0:
        assert int(self.args.num_evaluation_tasks / self.batch_size) % 5 == 0  # make sure the last iter is 0
        if self.val_iter % 5 != 0:
            reduce = False      # 所有val完了，所有的一起kmeans
            flag = f"{flag}_medium"
        assigns, gates, cluster_idxs, scores = self.pool_put(
            data_batch, true_labels, pool=self.val_pool, reduce=reduce, dataset_idxs=dataset_idxs, flag=flag)

        if self.val_iter % int(self.args.num_evaluation_tasks / self.batch_size) == 0:
            # iter % 125 == 0: this is the last batch in this epoch
            ## 如果是每个epoch的最后一个batch，额外跑val_pool_forward,
            ## 随机选2个clusters中随机选5个class组成tasks当evaluation tasks
            ## 8个clusters随机选4个中随机选5个class组成tasks当population

            # check num of available queue
            pool_flag = self.check_pool(pool=self.val_pool, max=self.args.num_classes_per_set)  # self.args.num_classes_per_set * 2
            num_obj, num_pop, num_mix = 2, 2, 2  # num_obj, num_pop, num_mix = 8, 8, 2
            assert num_mix == 2         # mix策略是用Beta(5,2)和Beta(2,5)
            # if flag is 'after' and pool_flag:       # during the training, we do not calculate conflict loss.
            if pool_flag:
                ## repeat the following process several times, and get the mean loss
                ## conflict_losses 就是 {}_loss_{} {}_accuracy_{}, 直接每次repeat之后的到的conflict_losses每个key的value都collect最后平均。

                conflict_losses = dict()
                HV_loss = []
                HV_acc = []
                for cflct_iter in range(50):
                    structured_sampled_data_batch = self.pool_get(
                        num_obj=num_obj, num_pop=num_pop, num_mix=num_mix, pool=self.val_pool)
                    # used to calculate conflict_losses
                    # [[_c1_][_c2_][_c3_][_c4_][_c5_][_c6_][_c7_][_c8_]
                    # [c1][c2][c3][c4][c5][c6][c7][c8][mix11][mix12][mix21][mix22]]

                    x_support_set, x_target_set, y_support_set, y_target_set = structured_sampled_data_batch

                    x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
                    x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
                    if hasattr(self.args, "is_regression") and self.args.is_regression:
                        y_support_set = torch.Tensor(y_support_set).float().to(device=self.device)
                        y_target_set = torch.Tensor(y_target_set).float().to(device=self.device)
                    else:
                        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
                        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)
                    # x_support_set [20, n_way, shot, c, h, w]
                    # x_target_set  [8, n_way, Kq, c, h, w]
                    # y_support_set [20, n_way, shot]
                    # y_target_set  [8, n_way, Kq]

                    data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

                    # get conflict losses as 4 theta_is on task [[c1][c2]]
                    # conflict losses: [4, 2] on loss and accuracy
                    conflict_losses_i, _ = self.pool_forward(
                        data_batch=data_batch, epoch=self.current_epoch,
                        num_obj=num_obj, num_pop=num_pop, num_mix=num_mix, val=True)
                    for key, value in conflict_losses_i.items():
                        if key in conflict_losses:
                            conflict_losses[key].append(value)
                        else:
                            conflict_losses[key] = [value]

                    ## cal val HV loss
                    # obtain pop_objs for conflict loss
                    pop_objs = []
                    for pop_idx in range(num_pop + num_mix):
                        # 4
                        objs = []
                        for obj_idx in range(num_obj):
                            obj = np.mean(np.stack(
                                [conflict_losses_i['{}_loss_{}'.format(pop_idx, item_idx)] for item_idx in
                                 range(obj_idx * self.args.num_sampled_tasks_for_each_objective,
                                       (obj_idx + 1) * self.args.num_sampled_tasks_for_each_objective)]))
                            objs.append(obj)
                        objs = np.stack(objs)
                        pop_objs.append(objs)  # [pop_size, obj_size]
                    pop_objs = np.stack(pop_objs)  # [4, 2]
                    HV_loss.append(cal_HV_loss(pop_objs) if self.args.HV_weight != 0.0 else 0)

                    ## cal val HV accuracy
                    # obtain pop_objs for conflict accuracy
                    pop_objs = []
                    for pop_idx in range(num_pop + num_mix):
                        # 4
                        objs = []
                        for obj_idx in range(num_obj):
                            obj = np.mean(np.stack(
                                [conflict_losses_i['{}_accuracy_{}'.format(pop_idx, item_idx)] for item_idx in
                                 range(obj_idx * self.args.num_sampled_tasks_for_each_objective,
                                       (obj_idx + 1) * self.args.num_sampled_tasks_for_each_objective)]))
                            objs.append(obj)
                        objs = np.stack(objs)
                        pop_objs.append(objs)  # [pop_size, obj_size]
                    pop_objs = np.stack(pop_objs)  # [4, 2]
                    HV_acc.append(cal_HV_acc(pop_objs) if self.args.HV_weight != 0.0 else 0)

                for key, value_list in conflict_losses.items():
                    conflict_losses[key] = np.mean(value_list)
                HV_loss = np.mean(HV_loss)
                HV_acc = np.mean(HV_acc)

            else:       # zero conflict loss
                conflict_losses = self.get_zero_conflict_losses(
                    obj_size=num_obj,
                    pop_size=num_pop + num_mix)
                HV_loss = 0
                HV_acc = 0
            conflict_losses['HV_loss'] = HV_loss
            conflict_losses['HV_acc'] = HV_acc
            losses = self.build_prefix_dict(conflict_losses, 'conflict', prefixed_losses=losses)

            # reset val_iter
            self.val_iter = 0

            # return losses, per_task_target_preds, {
            #     'adapted_weights_batch': adapted_weights_batch, 'medium_batch': medium_batch,
            #     'conflict_losses': conflict_losses}

        return losses, per_task_target_preds, {
            'adapted_weights_batch': adapted_weights_batch, 'medium_batch': medium_batch,
            'assigns': assigns, 'gates': gates, 'cluster_idxs': cluster_idxs, 'scores': scores}

    @torch.no_grad()
    def debug_collect_class_clustering_info(self, data_batch):
        """
        used to debug
        task to cls instances obtain clustering info
        Args:
            data_batch:

        Returns:

        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch
        # [batch_size, n_way, shot/Kq, c, h, w]; [batch_size, n_way, shot/Kq]
        # class: [shot/Kq, c, h, w]; [shot/Kq]
        bs, n, s, c, h, w = x_support_set.shape
        _, _, kq, _, _, _ = x_target_set.shape
        cluster_layer_0 = self.args.cluster_layer_0
        cluster_layer_1 = self.args.cluster_layer_1 if self.args.cluster_layer_1 > 0 else 1

        assert not x_support_set.is_cuda

        assigns, gates, cluster_idxs, scores = [], [], [], []
        # for each class instance   bs*N
        for task_idx in range(bs):
            for n_idx in range(n):
                class_batch = (x_support_set[task_idx, n_idx].unsqueeze(dim=0),
                               x_target_set[task_idx, n_idx].unsqueeze(dim=0))
                # [1, s/kq, c, h, w]
                _, class_emb_vec = self.class_batch_2_class_embs(class_batch, n)   # [1, 1, 128]
                class_emb_vec = class_emb_vec[0]  # [1, 128]

                assert class_emb_vec.shape[0] == 1

                assign, gate, cluster_idx, score = \
                    self.taskSpecificNet.forward_task_emb_get_clustering_information(class_emb_vec)
                assigns.append(assign)
                gates.append(gate)
                cluster_idxs.append(cluster_idx)
                scores.append(score)

        assigns = np.concatenate(assigns)               # [bs*n, 4]
        gates = np.concatenate(gates)                   # [bs*n, 4, 2]
        cluster_idxs = np.concatenate(cluster_idxs)     # [bs*n]
        scores = np.concatenate(scores)                 # [bs*n, 8]

        assigns = assigns.reshape((bs, n, cluster_layer_0))
        gates = gates.reshape((bs, n, cluster_layer_0, cluster_layer_1))
        cluster_idxs = cluster_idxs.reshape((bs, n))
        scores = scores.reshape((bs, n, cluster_layer_0 * cluster_layer_1))

        return assigns, gates, cluster_idxs, scores


def pearson_correlation_coefficient(objs):
    """
    objs: [pop_size, 2]
    calculate PCC only for obj_size 2
    Args:
        objs: population with shape(pop_size, 2)

    Returns: pearson (Tensor)

    """
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    pearson = cos(objs[:, 0] - torch.mean(objs[:, 0]), objs[:, 1] - torch.mean(objs[:, 1]))
    return pearson


def cal_lp_loss(objs, p):
    """
    objs: [pop_size, obj_size]
    calculate lp loss as:
    first normalization
    \sum_pop (\sum_obj (obj)^p - 1)^2
    Args:
        objs: population with shape(pop_size, 2)
        p:
    Returns: loss (Tensor)

    """
    # [0,1] normalization
    objs = objs - objs.min(0, keepdim=True)[0]
    objs = objs / (objs.max(0, keepdim=True)[0] + 1e-6)

    loss = torch.mean(torch.stack([
        (torch.sum(torch.stack([
            objs[pop_idx, obj_idx] ** p for obj_idx in range(objs.shape[1])]
        )) - 1)     # ** 2
        for pop_idx in range(objs.shape[0])]))
    return loss


def cal_diversity_loss(objs):
    """
    objs: [pop_size, obj_size]
    calculate diversity loss as:
    first normalization
    2 point with min distance d, loss = -log(d)
    Args:
        objs: population with shape(pop_size, 2)
    Returns: loss (Tensor)

    """
    # [0,1] normalization
    objs = objs - objs.min(0, keepdim=True)[0]
    objs = objs / (objs.max(0, keepdim=True)[0] + 1e-6)

    # find min distance d
    pop_size, obj_size = objs.shape
    min_dist = None
    for indi1_idx in range(pop_size-1):
        for indi2_idx in range(indi1_idx+1, pop_size):
            dist = torch.dist(objs[indi1_idx], objs[indi2_idx], p=2)
            if min_dist is None or min_dist > dist:
                min_dist = dist
    return -torch.log(min_dist)


def cal_IGD_loss(objs, refs):
    """
    pymoo -> reference vector -> scale to \sum f1**p = 1. Then calculate IGD
    # generate referenece points
    refs = get_reference_directions(
        "energy", self.obj_size, args.pop_size, seed=1, scaling=0.9)  # , scaling=0.5
    Args:
        objs: population with shape(pop_size, 2)
        refs: set of reference points (num_refs, 2)

    Returns: loss (Tensor)
    """
    # [0,1] normalization
    objs = objs - objs.min(0, keepdim=True)[0]
    objs = objs / (objs.max(0, keepdim=True)[0] + 1e-6)

    # non-dominated selection
    if type(objs) is torch.Tensor:
        objs_np = objs.detach().cpu().numpy()
    else:
        objs_np = objs
    nds = NonDominatedSorting()
    fronts = nds.do(objs_np)
    nd_idxs = fronts[0]
    objs = objs[nd_idxs]    # [num_nd, 2]

    # for each reference point, calculate the minimal distance to the population, then sum
    # loss = []
    # for ref in refs:
    #     correspond_obj = objs[
    #                    torch.argmin(torch.stack([
    #                        torch.dist(ref, indi, p=2) for indi in objs
    #                    ]))
    #                ]
    #     # loss.append(torch.sum(correspond_obj - ref))    # manhattan dist to encourage negative
    #     loss.append(torch.sqrt(torch.sum(torch.square(correspond_obj - ref))))
    #
    # loss = torch.mean(torch.stack(loss))

    # loss = torch.mean(torch.stack([
    #     torch.dist(ref,
    #                objs[
    #                    torch.argmin(torch.stack([
    #                        torch.dist(ref, indi, p=2) for indi in objs
    #                    ]))
    #                ]) for ref in refs
    # ]))

    loss = []
    for ref in refs:
        min_dist = None
        for indi in objs:
            dist = torch.dist(ref, indi, p=2)
            if min_dist is None or min_dist > dist:
                min_dist = dist
        loss.append(min_dist)

    loss = torch.mean(torch.stack(loss))
    return loss


def cal_GD_loss(objs, refs):
    """
    objs: [pop_size, obj_size]
    calculate GD loss
    first normalization
    Args:
        objs: population with shape(pop_size, 2)
        refs: set of reference points (num_refs, 2)
    Returns: loss (Tensor)

    """
    # [0,1] normalization
    objs = objs - objs.min(0, keepdim=True)[0]
    objs = objs / (objs.max(0, keepdim=True)[0] + 1e-6)

    # non-dominated selection
    if type(objs) is torch.Tensor:
        objs_np = objs.detach().cpu().numpy()
    else:
        objs_np = objs
    nds = NonDominatedSorting()
    fronts = nds.do(objs_np)
    nd_idxs = fronts[0]
    objs = objs[nd_idxs]    # [num_nd, 2]

    # for each reference point, calculate the minimal distance to the population, then sum
    # loss = []
    # for indi in objs:
    #     correspond_ref = refs[
    #                    torch.argmin(torch.stack([
    #                        torch.dist(ref, indi, p=2) for ref in refs
    #                    ]))
    #                ]
    #     # loss.append(torch.sum(indi - correspond_ref))    # manhattan dist to encourage negative
    #     # loss.append(torch.dist(indi, correspond_ref, p=2))
    #     loss.append(torch.sqrt(torch.sum(torch.square(indi - correspond_ref))))
    #
    # loss = torch.mean(torch.stack(loss))

    # loss = torch.mean(torch.stack([
    #     torch.dist(indi,
    #                refs[
    #                    torch.argmin(torch.stack([
    #                        torch.dist(ref, indi, p=2) for ref in refs
    #                    ]))
    #                ], p=2) for indi in objs
    # ]))

    loss = []
    for indi in objs:
        min_dist = None
        for ref in refs:
            dist = torch.dist(ref, indi, p=2)
            if min_dist is None or min_dist > dist:
                min_dist = dist
        loss.append(min_dist)

    loss = torch.mean(torch.stack(loss))
    return loss


def cal_WS_loss(objs, refs):
    """
    objs: [pop_size, obj_size]
    calculate weighted sum loss
    first sort 4 objs based on [:, 0] since the order of these points may change during the iteration.
    ascending order, since refs is
    tensor([[0.0000, 1.0000],
            [0.3327, 0.6673],
            [0.6677, 0.3323],
            [1.0000, 0.0000]])
    Args:
        objs: population with shape(pop_size, 2)
        refs: set of reference vectors (pop_size, 2), should have same shape with objs
    Returns: loss (Tensor)

    """
    # sort
    _, indices = torch.sort(objs[:, 0])     # ascending order of x axis.
    objs = objs[indices]

    # loss
    loss = torch.sum(objs * refs) / objs.shape[0]

    return loss


def cal_entropy_loss(assign):
    """
    - 1/len(assign)  * \sum_i assign(i) log(assign(i))
    Args:
        assign: [4] or [bs, 4] or [bs, 4, 5]  [0,1]^4

    Returns: loss

    """
    assign = assign.view(-1)
    loss = - torch.sum(assign * torch.log(assign)) / assign.shape[0]

    return loss


def cal_HV_loss(objs):
    """
    HV calculation for 2-d

    normalization

    non-dominated selection

    Args:
        objs: population tensor with shape(pop_size, 2)     (4,2)

    Returns: loss

    """
    assert objs.shape[1] == 2

    # [0,1] normalization
    if type(objs) is torch.Tensor:
        objs = objs - objs.min(0, keepdim=True)[0]
        objs = objs / (objs.max(0, keepdim=True)[0] + 1e-6)
    else:
        objs = objs - objs.min(0)
        objs = objs / (objs.max(0) + 1e-6)

    # non-dominated selection
    if type(objs) is torch.Tensor:
        objs_np = objs.detach().cpu().numpy()
    else:
        objs_np = objs
    nds = NonDominatedSorting()
    fronts = nds.do(objs_np)
    nd_idxs = fronts[0]
    objs = objs[nd_idxs]    # [num_nd, 2]

    # sort through f1
    if type(objs) is torch.Tensor:
        _, idxs = objs[:, 0].sort()
        objs = objs[idxs]       # ascending order through f1
    else:
        idxs = np.argsort(objs[:, 0])
        objs = objs[idxs]       # ascending order through f1

    # update objs_np and objs to nd set.
    if type(objs) is torch.Tensor:
        objs_np = objs.detach().cpu().numpy()
    else:
        objs_np = objs

    # HV cal
    # max_objs = np.max(objs_np, axis=0)      # [2,]
    # min_objs = np.min(objs_np, axis=0)      # [2,]
    # ref_np = (max_objs - min_objs) * 1.1 + min_objs
    # ref_np = max_objs * 1.1     # np.array([2.0, 2.0])      # max_objs * 1.5
    ref_np = np.array([1.5, 1.5])

    # normalize objs

    if type(objs) is torch.Tensor:
        ref = torch.from_numpy(ref_np).to(objs.device)
    else:
        ref = ref_np

    hv = []
    num_nd = objs.shape[0]
    for idx in range(num_nd - 1):
        hv.append((objs[idx+1, 0] - objs[idx, 0]) * (ref[1] - objs[idx, 1]))
    hv.append((ref[0] - objs[num_nd-1, 0]) * (ref[1] - objs[num_nd-1, 1]))

    if type(objs) is torch.Tensor:
        hv = torch.stack(hv)
        hv = torch.sum(hv)
    else:
        hv = np.stack(hv)
        hv = np.sum(hv)

    return -hv


def cal_HV_acc(objs):
    """
    HV calculation for 2-d

    normalization

    non-dominated selection

    reference point [0, 0]

    Args:
        objs: population tensor with shape(pop_size, 2)     (4,2)

    Returns: acc

    """
    assert objs.shape[1] == 2

    # # [0,1] normalization
    # objs = objs - objs.min(0, keepdim=True)[0]
    # objs = objs / (objs.max(0, keepdim=True)[0] + 1e-6)

    # non-dominated selection
    if type(objs) is torch.Tensor:
        objs_np = objs.detach().cpu().numpy()
    else:
        objs_np = objs
    nds = NonDominatedSorting()
    fronts = nds.do(objs_np)
    nd_idxs = fronts[0]
    objs = objs[nd_idxs]    # [num_nd, 2]

    # sort through f1
    if type(objs) is torch.Tensor:
        _, idxs = objs[:, 0].sort()
        objs = objs[idxs]       # ascending order through f1
    else:
        idxs = np.argsort(objs[:, 0])
        objs = objs[idxs]       # ascending order through f1

    # HV cal
    hv = []
    num_nd = objs.shape[0]
    hv.append(objs[0, 0] * objs[0, 1])

    for idx in range(1, num_nd):
        hv.append((objs[idx, 0] - objs[idx-1, 0]) * objs[idx, 1])

    if type(objs) is torch.Tensor:
        hv = torch.stack(hv)
        hv = torch.sum(hv)
    else:
        hv = np.stack(hv)
        hv = np.sum(hv)

    return hv
