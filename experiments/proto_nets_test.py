"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet, MultiDataset, Meta
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.callbacks import *
from few_shot.eval import evaluate_with_fn
from few_shot.utils import setup_dirs
from config import PATH


setup_dirs()

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0)
parser.add_argument('--dataset')
parser.add_argument('--test-dataset')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=60, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)

args = parser.parse_args()

assert torch.cuda.is_available()
torch.cuda.set_device(int(args.device))
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

evaluation_episodes = 1000
episodes_per_epoch = 100

n_epochs = 100
dataset_class = Meta
num_input_channels = 3
drop_lr_every = 40

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

print(param_str)

###################
# Create datasets #
###################

evaluation = dataset_class('evaluation', args.test_dataset)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)


#########
# Model #
#########
filepath = PATH + f'/models/proto_nets/{param_str}.pth'
model = get_few_shot_encoder(num_input_channels)
model.load_state_dict(torch.load(filepath))

model.to(device, dtype=torch.double)


############
# testing  #
############
print(f'Testing Prototypical network on {args.test_dataset}...')
# loss_fn = torch.nn.NLLLoss().cuda()

val_categorical_accuracies = []
for run in range(20):
    logs = evaluate_with_fn(
        model,
        dataloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        metrics=['categorical_accuracy'],
        loss_fn=torch.nn.NLLLoss().cuda(),
        eval_fn=proto_net_episode,
        eval_function_kwargs={'n_shot': args.n_test, 'k_way': args.k_test, 'q_queries': args.q_test,
                              'distance': args.distance}
    )
    # print(logs)
    val_categorical_accuracies.append(logs['val_categorical_accuracy'])

# statistic
print(f'val_categorical_accuracies: {val_categorical_accuracies}')
print(f'mean accuracy: {np.mean(val_categorical_accuracies)}, std: {np.std(val_categorical_accuracies)}')
