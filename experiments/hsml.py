import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
import argparse
import platform

from few_shot.datasets import OmniglotDataset, MiniImageNet, MultiDataset, Meta
from few_shot.core import NShotTaskSamplerMultiDomain, create_nshot_task_label, EvaluateFewShot
from few_shot.hsml.hsml import HSML
from few_shot.hsml.utils import HSMLCheckpoint
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH


setup_dirs()

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', default='BTAF-hsml')
# dataset
parser.add_argument('--dataset', default='BTAF')
parser.add_argument('--num-input-channels', default=3, type=int)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--num-classes-per-set', default=5, type=int, help='k-way')
parser.add_argument('--num-samples-per-class', default=2, type=int, help='n-shot')
parser.add_argument('--num-target-samples', default=3, type=int, help='q-query')
parser.add_argument('--seed', default=9, type=int, help='seed to fix randomization')
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
parser.add_argument('--aug-weight', default=0.01, type=float, help='only activate when use-augment')
parser.add_argument('--HV-weight', default=0.01, type=float, help='HV loss weight')
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--epoch-len', default=10, type=int)
parser.add_argument('--eval-batches', default=5, type=int)
parser.add_argument('--use-warm-start', action='store_true', help='true to load checkpoint')
args = parser.parse_args()

args.experiment_name = f'{args.experiment_name}-{args.seed}'

assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

if args.dataset == 'BTAF':
    args.dataset_name = ["meta_CUB_Bird", "meta_FGVC_Aircraft", "meta_DTD_Texture", "meta_FGVCx_Fungi"]
    args.is_regression = False
    args.final_layer_size = 1200
    args.number_filters = 48
    args.num_input_channels = 3
    args.image_height = 84
    args.image_width = 84
    dataset_class = MultiDataset
elif args.dataset == 'meta':
    args.dataset_name = [
        "meta_CUB_Bird", "meta_FGVC_Aircraft", "meta_DTD_Texture", "meta_FGVCx_Fungi",
        "Omniglot_84", "VGG_Flower_84", "mini_84", "quickdraw_84"]
    args.is_regression = False
    args.final_layer_size = 1200
    args.number_filters = 48
    args.num_input_channels = 3
    args.image_height = 84
    args.image_width = 84
    dataset_class = MultiDataset
elif args.dataset == 'miniImageNet':
    args.dataset_name = ["MiniImageNet"]
    args.is_regression = False
    args.final_layer_size = 1200
    args.number_filters = 48
    args.num_input_channels = 3
    args.image_height = 84
    args.image_width = 84
    dataset_class = MiniImageNet
else:
    raise Exception('training dataset setting not implement, please use BTAF')

print(args)

###################
# Create datasets #
###################
preload = True
num_workers = 8     # 8
if platform.system() == 'Windows':
    num_workers = 0
if args.dataset == 'BTAF':
    background = dataset_class([Meta('background', 'CUB_Bird', preload=preload),
                                Meta('background', 'DTD_Texture', preload=preload),
                                Meta('background', 'FGVC_Aircraft', preload=preload),
                                Meta('background', 'FGVCx_Fungi', preload=preload)])
elif args.dataset == 'meta':
    background = dataset_class([Meta('background', 'CUB_Bird', preload=preload),
                                Meta('background', 'DTD_Texture', preload=preload),
                                Meta('background', 'FGVC_Aircraft', preload=preload),
                                Meta('background', 'FGVCx_Fungi', preload=preload),
                                Meta('background', 'Omniglot_84', preload=preload),
                                Meta('background', 'VGG_Flower_84', preload=preload),
                                Meta('background', 'mini_84', preload=preload),
                                Meta('background', 'quickdraw_84', preload=preload)
                                ])
else:
    background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSamplerMultiDomain(background, args.epoch_len,
                                              n=args.num_samples_per_class, k=args.num_classes_per_set,
                                              q=args.num_target_samples,
                                              num_tasks=args.batch_size),
    num_workers=num_workers
)

if args.dataset == 'BTAF':
    evaluation = dataset_class([Meta('evaluation', 'CUB_Bird', preload=preload),
                                Meta('evaluation', 'DTD_Texture', preload=preload),
                                Meta('evaluation', 'FGVC_Aircraft', preload=preload),
                                Meta('evaluation', 'FGVCx_Fungi', preload=preload)])
elif args.dataset == 'meta':
    evaluation = dataset_class([Meta('evaluation', 'CUB_Bird', preload=preload),
                                Meta('evaluation', 'DTD_Texture', preload=preload),
                                Meta('evaluation', 'FGVC_Aircraft', preload=preload),
                                Meta('evaluation', 'FGVCx_Fungi', preload=preload),
                                Meta('evaluation', 'Omniglot_84', preload=preload),
                                Meta('evaluation', 'VGG_Flower_84', preload=preload),
                                Meta('evaluation', 'mini_84', preload=preload),
                                Meta('evaluation', 'quickdraw_84', preload=preload)
                                ])
else:
    evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSamplerMultiDomain(evaluation, args.eval_batches,
                                              n=args.num_samples_per_class, k=args.num_classes_per_set,
                                              q=args.num_target_samples,
                                              num_tasks=args.batch_size),
    num_workers=num_workers
)

#########
# Model #
#########
print(f'Training HSML on {args.dataset}...')
np.random.seed(seed=42)
torch.manual_seed(seed=42)
meta_model = HSML(args, device).to(device)  # , dtype=torch.double
loss_fn = nn.CrossEntropyLoss().to(device)


############
# Training #
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


debug = False

if not debug:
    callbacks = [
        Seeder(seed=args.seed, num_epochs=args.epochs),
        EvaluateFewShot(
            eval_fn=meta_model.run_test_iter,
            num_tasks=args.eval_batches,
            n_shot=args.num_samples_per_class,
            k_way=args.num_classes_per_set,
            q_queries=args.num_target_samples,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_meta_batch(
                args.num_samples_per_class, args.num_classes_per_set, args.num_target_samples, args.batch_size),
            # eval_fn kwargs
            medium_record=True,     # False when run
        ),
        HSMLCheckpoint(
            filepath=PATH + f'/models/hsml/{args.experiment_name}.pth',
            monitor=f'val_{args.num_samples_per_class}-shot_{args.num_classes_per_set}-way_acc',
            verbose=0, save_best_only=False, load=args.use_warm_start
        ),
        ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
        # it may influence warm start. and will introduce 'lr' key in epoch_logs.
        CSVLogger(PATH + f'/logs/hsml/{args.experiment_name}.csv',
                  # keys=['val_loss', f'val_{args.num_samples_per_class}-shot_{args.num_classes_per_set}-way_acc'],
                  append=True if args.use_warm_start else False
                  ),
    ]

    fit(
        meta_model,
        meta_model.optimizer,   # no use
        loss_fn,                # no use
        epochs=args.epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_meta_batch(
            args.num_samples_per_class, args.num_classes_per_set, args.num_target_samples, args.batch_size),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        verbose=True,
        fit_function=meta_model.run_train_iter,
        fit_function_kwargs={'medium_record': False},
    )

else:       # debug
    for i, batch in enumerate(evaluation_taskloader):
        if i==0:
            print(batch[0].shape, batch[1].shape)
            # torch.Size([100, 3, 84, 84]) torch.Size([100])
            # bs[4]*(n[2]*s[5]+q[3]*s)[25]=100

    x, y = batch
    x = x.reshape(4, 2*5+3*5, 3, 84, 84)
    y = y.reshape(4, 2*5+3*5)
    print(y[0, :20])
    # tensor([26, 26, 34, 34, 13, 13, 44, 44, 48, 48, 26, 26, 26, 34, 34, 34, 13, 13,
    #         13, 44])

    x_support_set = x[:, :2*5]
    y_support_set = y[:, :2*5]
    x_support_set = x_support_set.reshape(4, 5, 2, 3, 84, 84)
    y_support_set = y_support_set.reshape(4, 5, 2)
    print(y_support_set[0, :, :])
    # tensor([[26, 26],
    #         [34, 34],
    #         [13, 13],
    #         [44, 44],
    #         [48, 48]])
    y_target_set = y[:, 2*5:].reshape(4, 5, 3)
    print(y_target_set[0, :, :])
    # tensor([[26, 26, 26],
    #         [34, 34, 34],
    #         [13, 13, 13],
    #         [44, 44, 44],
    #         [48, 48, 48]])

    prepare_batch = prepare_meta_batch(
        args.num_samples_per_class, args.num_classes_per_set, args.num_target_samples, args.batch_size)
    x, y = prepare_batch((x, y))
    data_batch_cuda, true_labels = meta_model.batch_reshape(x, y)
    # data_batch_cuda:
    # torch.Size([4, 5, 2, 3, 84, 84])
    # torch.Size([4, 5, 3, 3, 84, 84])
    # torch.Size([4, 5, 2])
    # torch.Size([4, 5, 3])
    # true_labels
    # tensor([[20, 18, 69, 17, 40],
    #         [24,  8, 54, 44, 53],
    #         [28, 36, 41, 51, 54],
    #         [29, 31, 65, 63, 27]])
    x_cuda, y_cuda = meta_model.batch_reshape_back(data_batch_cuda)
