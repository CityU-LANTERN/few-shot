"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
from torch.utils.data import DataLoader
from torch import nn
import argparse
import platform

from few_shot.datasets import OmniglotDataset, MiniImageNet, MultiDataset, Meta
from few_shot.core import NShotTaskSamplerMultiDomain, create_nshot_task_label, EvaluateFewShot
from few_shot.maml import meta_gradient_step
from few_shot.models import FewShotClassifier
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH


setup_dirs()

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--seed', default=9, type=int, help='seed to fix randomization')
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)  # Number of examples per class to calculate meta gradients with
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--inner-val-steps', default=3, type=int)
parser.add_argument('--inner-lr', default=0.4, type=float)
parser.add_argument('--meta-lr', default=0.001, type=float)
parser.add_argument('--meta-batch-size', default=32, type=int)
parser.add_argument('--order', default=2, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch-len', default=100, type=int)
parser.add_argument('--eval-batches', default=20, type=int)

args = parser.parse_args()

assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

if args.dataset == 'omniglot':
    dataset_class = OmniglotDataset
    fc_layer_size = 64
    number_filters = 64
    num_input_channels = 1
elif args.dataset == 'miniImageNet':
    dataset_class = MiniImageNet
    fc_layer_size = 1200            # 1600 for filter64, 1200 for filter48
    number_filters = 48
    num_input_channels = 3
elif args.dataset == 'BTAF':
    dataset_class = MultiDataset
    fc_layer_size = 1200
    number_filters = 48
    num_input_channels = 3
else:
    raise(ValueError('Unsupported dataset'))

param_str = f'{args.dataset}_order={args.order}_n={args.n}_k={args.k}_metabatch={args.meta_batch_size}_' \
            f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}_seed={args.seed}'
print(param_str)


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
else:
    background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSamplerMultiDomain(background, args.epoch_len, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=num_workers
)

if args.dataset == 'BTAF':
    evaluation = dataset_class([Meta('evaluation', 'CUB_Bird', preload=preload),
                                Meta('evaluation', 'DTD_Texture', preload=preload),
                                Meta('evaluation', 'FGVC_Aircraft', preload=preload),
                                Meta('evaluation', 'FGVCx_Fungi', preload=preload)])
else:
    evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSamplerMultiDomain(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=num_workers
)


############
# Training #
############
print(f'Training MAML on {args.dataset}...')
np.random.seed(seed=42)
torch.manual_seed(seed=42)
meta_model = FewShotClassifier(num_input_channels, args.k, fc_layer_size, number_filters).to(device)
# .to(device, dtype=torch.double)
meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
loss_fn = nn.CrossEntropyLoss().to(device)


def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
        # x = x.double().to(device)
        x = x.float().to(device)
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_


callbacks = [
    Seeder(seed=args.seed, num_epochs=args.epochs),
    EvaluateFewShot(
        eval_fn=meta_gradient_step,
        num_tasks=args.eval_batches,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
        # MAML kwargs
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        device=device,
        order=args.order,
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/maml/{param_str}.pth',
        monitor=f'val_{args.n}-shot_{args.k}-way_acc'
    ),
    ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
    CSVLogger(PATH + f'/logs/maml/{param_str}.csv'),
]


fit(
    meta_model,
    meta_optimiser,
    loss_fn,
    epochs=args.epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=meta_gradient_step,
    fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                         'train': True,
                         'order': args.order, 'device': device, 'inner_train_steps': args.inner_train_steps,
                         'inner_lr': args.inner_lr},
)
