import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union

from few_shot.metrics import NAMED_METRICS


def evaluate(model: Module, dataloader: DataLoader, prepare_batch: Callable, metrics: List[Union[str, Callable]],
             loss_fn: Callable = None, prefix: str = 'val_', suffix: str = ''):
    """Evaluate a model on one or more metrics on a particular dataset

    # Arguments
        model: Model to evaluate
        dataloader: Instance of torch.utils.data.DataLoader representing the dataset
        prepare_batch: Callable to perform any desired preprocessing
        metrics: List of metrics to evaluate the model with. Metrics must either be a named metric (see `metrics.py`) or
            a Callable that takes predictions and ground truth labels and returns a scalar value
        loss_fn: Loss function to calculate over the dataset
        prefix: Prefix to prepend to the name of each metric - used to identify the dataset. Defaults to 'val_' as
            it is typical to evaluate on a held-out validation dataset
        suffix: Suffix to append to the name of each metric.
    """
    logs = {}
    seen = 0
    totals = {m: 0 for m in metrics}
    if loss_fn is not None:
        totals['loss'] = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = prepare_batch(batch)
            y_pred = model(x)

            seen += x.shape[0]

            if loss_fn is not None:
                totals['loss'] += loss_fn(y_pred, y).item() * x.shape[0]

            for m in metrics:
                if isinstance(m, str):
                    v = NAMED_METRICS[m](y, y_pred)
                else:
                    # Assume metric is a callable function
                    v = m(y, y_pred)

                totals[m] += v * x.shape[0]

    for m in ['loss'] + metrics:
        logs[prefix + m + suffix] = totals[m] / seen

    return logs


def evaluate_with_fn(model: Module, dataloader: DataLoader, prepare_batch: Callable,
                     metrics: List[Union[str, Callable]],
                     eval_fn: Callable, eval_function_kwargs: dict, loss_fn: Callable = None,
                     prefix: str = 'val_', suffix: str = ''):
    """Evaluate a model on one or more metrics on a particular dataset with a perticular function

    # Arguments
        model: Model to evaluate
        dataloader: Instance of torch.utils.data.DataLoader representing the dataset
        prepare_batch: Callable to perform any desired preprocessing
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        eval_function_kwargs: parameters for eval_fn
        metrics: List of metrics to evaluate the model with. Metrics must either be a named metric (see `metrics.py`) or
            a Callable that takes predictions and ground truth labels and returns a scalar value
        loss_fn: Loss function to calculate over the dataset
        prefix: Prefix to prepend to the name of each metric - used to identify the dataset. Defaults to 'val_' as
            it is typical to evaluate on a held-out validation dataset
        suffix: Suffix to append to the name of each metric.
    """
    logs = {}
    seen = 0
    totals = {m: 0 for m in metrics}
    if loss_fn is not None:
        totals['loss'] = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = prepare_batch(batch)

            # print(f'x: {x.shape}, y: {y.shape}')

            y_pred = eval_fn(
                model,
                None,
                loss_fn,
                x,
                y,
                train=False,
                **eval_function_kwargs
            )
            # y_pred = model(x)

            seen += x.shape[0]

            if loss_fn is not None:
                totals['loss'] += loss_fn(y_pred, y).item() * x.shape[0]

            for m in metrics:
                if isinstance(m, str):
                    v = NAMED_METRICS[m](y, y_pred)
                else:
                    # Assume metric is a callable function
                    v = m(y, y_pred)

                totals[m] += v * x.shape[0]

    for m in ['loss'] + metrics:
        logs[prefix + m + suffix] = totals[m] / seen

    return logs
