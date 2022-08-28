"""
some callbacks.
"""
from typing import List, Iterable, Callable, Tuple
import numpy as np
import torch
import warnings
import os

from few_shot.callbacks import Callback


class HSMLCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options, which will be filled the value of `epoch` and keys in `logs`
    (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints will be saved
    with the epoch number and the validation loss in the filename.

    load checkpoint on train begin

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='auto', period=1, load=True):
        super(HSMLCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.load = load

        if mode not in ['auto', 'min', 'max']:
            raise ValueError('Mode must be one of (auto, min, max).')

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less

        self.best = np.Inf

    def on_train_begin(self, logs=None):
        """ load checkpoint, update self.best
            self.model is not None
        :param logs:
        :return:
        """
        if self.load and os.path.exists(self.filepath):
            state = torch.load(self.filepath)
            self.model.load_state_dict(state['network'])
            self.model.pool = state['pool']
            self.model.current_epoch = state['current_epoch']
            # needed for deciding whether to use conflict loss and the start epoch
            if self.monitor in state:
                self.best = state[self.monitor]
                # else use the current monitor and keep initial self.best
            # ignore state['args']
            if self.params['verbose']:
                print(f'{HSMLCheckpoint} '
                      f'load checkpoint from {self.filepath}, epoch start from {self.model.current_epoch}.')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # things need to store
        state = dict()
        state['network'] = self.model.state_dict()
        if 'val_medium_batch' in logs:
            state['val_medium_batch'] = logs['val_medium_batch']
        state['pool'] = self.model.pool
        state['args'] = self.model.args
        state['current_epoch'] = epoch
        # since it starts from 0, the next start epoch is this.
        state[self.monitor] = self.best

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            '''
            Option: store another [filename]_latest.pth file along with [filename]_[epoch].pth
            '''
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        torch.save(state, filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                torch.save(state, filepath)

    def on_batch_end(self, batch, logs=None):       # if need to store training medium batch, it needs to store in self.
        pass


def image_montage(X, imsize=None, maxw=10, minh=None):
    """X can be a list of images, or a matrix of vectorized images.
      Specify imsize when X is a matrix."""
    tmp = []
    numimgs = len(X)

    # create a list of images (reshape if necessary)
    for i in range(0, numimgs):
        if imsize != None:
            tmp.append(X[i].reshape(imsize))
        else:
            tmp.append(np.squeeze(X[i]))

    # add blanks
    if (numimgs > maxw) and (np.mod(numimgs, maxw) > 0):
        leftover = maxw - np.mod(numimgs, maxw)
        meanimg = 0.5 * (X[0].max() + X[0].min())
        for i in range(0, leftover):
            tmp.append(np.ones(tmp[0].shape) * meanimg)

    # add blank line
    if minh is not None and int(len(tmp) / maxw) < minh:
        leftover = (minh - int(len(tmp) / maxw)) * maxw
        meanimg = 0.5 * (X[0].max() + X[0].min())
        for i in range(0, leftover):
            tmp.append(np.ones(tmp[0].shape) * meanimg)

    # make the montage
    tmp2 = []
    for i in range(0, len(tmp), maxw):
        tmp2.append(np.hstack(tmp[i:i + maxw]))
    montimg = np.vstack(tmp2)
    return montimg


def task_montage(task_batch):
    # montage of the images in task
    x_support_set, x_target_set, _, _ = task_batch
    (b, n, k, c, h, w) = x_support_set.shape
    images = torch.cat((x_support_set, x_target_set), dim=2)  # [batch, N, K+Q, c, w,h]
    images = images.view(b, -1, c, h, w)
    task_image_montage_list = []
    for batch_id in range(b):
        task_image = images[batch_id].permute(0, 2, 3, 1).numpy()
        task_image_montage_list.append(image_montage(task_image))
    return image_montage(task_image_montage_list)


def pool_montage(pool, pool_size=16, pool_max_size=20):
    # montage of the images in pool
    queue_image_montage_list = []

    if type(pool) is dict:
        pool = pool['pool']

    for q in pool:
        image_batch = []
        for item in q:
            cl = item['class']
            if len(cl) == 2:
                x_support_set, x_target_set = cl
            else:
                x_support_set, x_target_set, _, _ = cl
            (k, c, h, w) = x_support_set.shape
            (t, _, _, _) = x_target_set.shape
            images = torch.cat((x_support_set, x_target_set), dim=0).permute(0, 2, 3, 1).numpy()  # [K+Q, w, h, c]
            image_batch.append(images)
        if len(image_batch) > 0:
            image_batch = np.concatenate(image_batch, axis=0)   # [(K+Q), w, h, c]
        else:
            image_batch = np.zeros((pool_max_size * 2 * (k + t), h, w, c))
        queue_image_montage_list.append(
            image_montage(image_batch,
                          maxw=int((k + t)/2),                      # 8
                          minh=pool_max_size * 2))        # 10*2

    return image_montage(queue_image_montage_list, maxw=pool_size)
