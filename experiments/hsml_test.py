import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
import argparse
import platform
from tqdm import tqdm

from few_shot.datasets import OmniglotDataset, MiniImageNet, MultiDataset, Meta
from few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from few_shot.hsml.hsml import HSML
from few_shot.hsml.utils import HSMLCheckpoint
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.eval import evaluate_with_fn
from few_shot.utils import setup_dirs
from config import PATH


setup_dirs()

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', default='BTAF-hsml')
# dataset
parser.add_argument('--test-dataset', default='CUB_Bird')
parser.add_argument('--num-input-channels', default=3, type=int)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--num-classes-per-set', default=5, type=int, help='k-way')
parser.add_argument('--num-samples-per-class', default=1, type=int, help='n-shot')
parser.add_argument('--num-target-samples', default=15, type=int, help='q-query')
parser.add_argument('--seed', default=10, type=int, help='seed to fix randomization')
# maml
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--inner-val-steps', default=3, type=int)
parser.add_argument('--order', default=2, type=int)
# TaskEmbedding
parser.add_argument('--hidden-dim', default=128, type=int)
parser.add_argument('--ae-type', default='gru', type=str)
# HierarchicalClustering
parser.add_argument('--cluster-layer-0', default=4, type=int)
parser.add_argument('--cluster-layer-1', default=4, type=int)
parser.add_argument('--sigma', default=10.0, type=float)
# TaskSpecific
parser.add_argument('--feature-extractor-only', action='store_true')
parser.add_argument('--use-spectral', action='store_true')
# pool
parser.add_argument('--use-pool', action='store_true', help='whether to use pool and obtain MO loss')
parser.add_argument('--use-conflict-loss', action='store_true', help='whether to use conflict loss to train')
parser.add_argument('--pool-start-epoch', default=0, type=int, help='when to start obtaining MO loss')  # 3
parser.add_argument('--pool-update-rate', default=1, type=int,
                    help='re-cluster the samples in the pool every # epochs')  # 2
parser.add_argument('--use-augment', action='store_true', help='True for Pool-Aug')
parser.add_argument('--pool-size', default=16, type=int, help='number of clusters in the pool')
parser.add_argument('--pool-max-size', default=20, type=int, help='number of class samples in a cluster')
parser.add_argument('--mix-method', default='cutmix', type=str, help='options: cutmix, mixup, mixupr')
parser.add_argument('--num-sampled-tasks-for-each-objective', default=1, type=int,
                    help='number of sampled tasks for each cluster')
# optimization
parser.add_argument('--meta-learning-rate', default=0.001, type=float, help='outer loop learning rate')
parser.add_argument('--inner-learning-rate', default=0.01, type=float, help='inner loop learning rate')
parser.add_argument('--clamp', default=10.0, type=float, help='clamp for gradients')
parser.add_argument('--emb-loss-weight', default=0.01, type=float, help='reconstruction loss weight')
args = parser.parse_args()

args.experiment_name = f'{args.experiment_name}-{args.seed}'

assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

args.dataset_name = [args.test_dataset]
args.is_regression = False
args.final_layer_size = 1200
args.number_filters = 48
args.num_input_channels = 3
args.image_height = 84
args.image_width = 84
dataset_class = Meta
evaluation_episodes = 1000

print(args)

###################
# Create datasets #
###################
preload = True
num_workers = 8     # 8
if platform.system() == 'Windows':
    num_workers = 0

evaluation = dataset_class('testing', args.test_dataset, preload=preload)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, evaluation_episodes,
                                   n=args.num_samples_per_class, k=args.num_classes_per_set,
                                   q=args.num_target_samples,
                                   num_tasks=1),
    num_workers=num_workers
)

#########
# Model #
#########
print(f'Testing HSML on {args.test_dataset}...')
if os.path.exists(PATH + f'/models/hsml/{args.experiment_name}-best.pth'):
    filepath = PATH + f'/models/hsml/{args.experiment_name}-best.pth'
else:
    filepath = PATH + f'/models/hsml/{args.experiment_name}.pth'
state = torch.load(filepath)
model = HSML(args, device).to(device)  # , dtype=torch.double
model.load_state_dict(state['network'])
# model.pool = state['pool']
# model.current_epoch = state['current_epoch']

np.random.seed(seed=42)
torch.manual_seed(seed=42)

############
# Testing  #
############
def prepare_meta_batch(n, k, q, meta_batch_size, to_cuda=True):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n*k + q*k, args.num_input_channels, x.shape[-2], x.shape[-1]).float()
        y = y.reshape(meta_batch_size, n * k + q * k)
        # Create label
        yq = create_nshot_task_label(k, q).repeat(meta_batch_size).reshape(meta_batch_size, -1)
        if hasattr(args, "is_regression") and args.is_regression:
            y = y.float()
        else:
            y = y.long()
        if to_cuda:
            # Move to device
            x = x.to(device)
            y = y.to(device)          # y is true labels
            yq = yq.to(device)        # yq is relative labels for query samples
            # labels

        return x, {'q_relative_labels': yq, 'true_labels': y}

    return prepare_meta_batch_

val_categorical_accuracies = []
for run in tqdm(range(20)):
    logs = evaluate_with_fn(
        model,
        dataloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(
            args.num_samples_per_class, args.num_classes_per_set, args.num_target_samples, meta_batch_size=1),
        metrics=['categorical_accuracy'],
        prefix='test_',
        loss_fn=model.loss_fn,
        eval_fn=model.run_test_iter,
        eval_function_kwargs={
            'n_shot': args.num_samples_per_class,
            'k_way': args.num_classes_per_set,
            'q_queries': args.num_target_samples,
            'epoch': 0, 'medium_record': False}
    )
    # print(logs)
    val_categorical_accuracies.append(logs['test_categorical_accuracy'])

# statistic
print(f'{args.test_dataset}: val_categorical_accuracies: {val_categorical_accuracies}')
print(f'mean accuracy: {np.mean(val_categorical_accuracies):.4f}, std: {np.std(val_categorical_accuracies):.4f}')
print('---------------------------------------------------------------------------------')
