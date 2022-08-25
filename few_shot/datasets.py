from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from typing import List, Dict

import sys
sys.path.append("..")
from config import DATA_PATH


class OmniglotDataset(Dataset):
    def __init__(self, subset, OOD_test=False):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the 'background' or 'evaluation' set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.OOD_test = OOD_test

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())     # [2636]
        # ['Angelic.0.character01', 'Angelic.0.character02', 'Angelic.0.character03', ...]

        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}     # {dict: 2636}
        # {'Angelic.0.character01': 0, 'Angelic.0.character02': 1, 'Angelic.0.character03': 2, ...}

        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))
        #     alphabet             class_name   filepath    subset     id class_id          {DataFrame: (52720, 6)}
        # 0  Angelic.0  Angelic.0.character01           ...             0        0
        # 1  Angelic.0  Angelic.0.character01           ...             1        0
        # 2  Angelic.0  Angelic.0.character01           ...             2        0
        # 3  Angelic.0  Angelic.0.character01           ...             3        0
        # 4  Angelic.0  Angelic.0.character01           ...             4        0

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']  # {dict: 52720}
        # {0: '//10.20.2.245/datasets/datasets/Omniglot_enriched/images_evaluation\\Angelic.0\\character01\\0965_01.png', ...}

        self.datasetid_to_class_id = self.df.to_dict()['class_id']  # {dict: 52720}
        # {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, ...}

        # Setup transforms      enable evaluation as OOD dataset
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),      # ToTensor() will normalize to [0, 1]
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, item):
        if self.OOD_test:
            instance = Image.open(self.datasetid_to_filepath[item])  # PNG, 28X28
            instance = instance.convert('RGB')
            instance = self.transform(instance)  # [3, 84, 84]
            label = self.datasetid_to_class_id[item]  # from 0 -> 20
            return instance, label
        else:
            instance = io.imread(self.datasetid_to_filepath[item])      # [28, 28]
            # Reindex to channels first format as supported by pytorch
            instance = instance[np.newaxis, :, :]                       # [1, 28, 28]

            # Normalise to 0-1
            instance = (instance - instance.min()) / (instance.max() - instance.min())

            label = self.datasetid_to_class_id[item]        # from 0 -> 2636

            return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot_enriched/images_{}'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot_enriched/images_{}'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split(os.sep)[-2]       # linux / ; windows \\
            # Angelic.0
            class_name = '{}.{}'.format(alphabet, root.split(os.sep)[-1])
            # Angelic.0.character01

            for f in files:
                if f.endswith('.png'):
                    progress_bar.update(1)
                    images.append({
                        'subset': subset,
                        'alphabet': alphabet,
                        'class_name': class_name,
                        'filepath': os.path.join(root, f)
                    })
            # filepath: //10.20.2.245/datasets/datasets/Omniglot_enriched/images_evaluation\\Angelic.0\\character01\\0965_01.png

        progress_bar.close()
        return images


class MiniImageNet(Dataset):
    def __init__(self, subset):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())     # [20]
        # ['n01770081', 'n02101006', 'n02108551', 'n02174001', 'n02219486', 'n02606052', 'n02747177', ...]

        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}       # {dict: 20}
        # {'n01770081': 0, 'n02101006': 1, 'n02108551': 2, 'n02174001': 3, 'n02219486': 4, ...}

        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))
        #   class_name   filepath    subset    id   class_id         {MiniImageNet: 12000}
        # 0  n01770081              ...         0          0
        # 1  n01770081              ...         1          0
        # 2  n01770081              ...         2          0
        # 3  n01770081              ...         3          0
        # 4  n01770081              ...         4          0

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']          # {dict: 12000}
        # {0: '//10.20.2.245/datasets/datasets/miniImageNet/images_evaluation\\n01770081\\00001098.jpg', ...}

        self.datasetid_to_class_id = self.df.to_dict()['class_id']          # {dict: 12000}
        # {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, ...}

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])     # JpegImageFile, 500x384
        instance = self.transform(instance)                         # [3, 84, 84]
        label = self.datasetid_to_class_id[item]                    # from 0 -> 20
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.jpg')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split(os.sep)[-1]     # linux / ; windows \\
            # n01770081

            for f in files:
                if f.endswith('.jpg'):
                    progress_bar.update(1)
                    images.append({
                        'subset': subset,
                        'class_name': class_name,
                        'filepath': os.path.join(root, f)
                    })
            # filepath: //10.20.2.245/datasets/datasets/miniImageNet/images_evaluation\\n01770081\\00001098.jpg

        progress_bar.close()
        return images


class Meta(Dataset):
    def __init__(self, subset, target, preload=False):
        """Dataset class for regular train/val/test,
        background -> train
        evaluation -> test

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
            target: which dataset to represent
            preload: whether to load whole dataset into memory
        """
        if subset not in ('background', 'evaluation', 'testing'):
            raise(ValueError, 'subset must be one of (background, evaluation, testing)')
        # if target not in ('CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi'):
        #     raise(ValueError, 'target must be one of (CUB_Bird, DTD_Texture, FGVC_Aircraft, FGVCx_Fungi)')
        self.subset = subset
        self.target = target
        self.preload = preload

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),      # ToTensor() will normalize to [0, 1]
        ])

        info_dict, self.memory = self.index_subset(self.subset, self.target, self.preload, self.transform)
        self.df = pd.DataFrame(info_dict)

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())     # [16]
        # ['014.Indigo_Bunting', '042.Vermilion_Flycatcher', '051.Horned_Grebe', ...]

        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}       # {dict: 16}
        # {'014.Indigo_Bunting': 0, '042.Vermilion_Flycatcher': 1, '051.Horned_Grebe': 2, ...}

        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))
        #   class_name            filepath    subset    id   class_id         {Bird: 960}
        # 0  014.Indigo_Bunting              ...         0          0
        # 1  014.Indigo_Bunting              ...         1          0
        # 2  014.Indigo_Bunting              ...         2          0
        # 3  014.Indigo_Bunting              ...         3          0
        # 4  014.Indigo_Bunting              ...         4          0

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']          # {dict: 960}
        # {0: '//10.20.2.245/datasets/datasets/meta-dataset/CUB_Bird/val\\014.Indigo_Bunting\\Indigo_Bunting_0001_12469.jpg', ...}

        self.datasetid_to_class_id = self.df.to_dict()['class_id']          # {dict: 960}
        # {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, ...}

    def __getitem__(self, item):
        if self.preload:
            instance = self.memory[item]
        else:
            instance = Image.open(self.datasetid_to_filepath[item])     # JpegImageFile, 84x84
            instance = self.transform(instance)                         # [3, 84, 84]
        label = self.datasetid_to_class_id[item]                        # from 0 -> 16
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset, target, preload=False, transform=None):
        """Index a subset by looping through all of its files and recording relevant information.
        if preload, store memory {Tensor: {num, 3, 84, 84}} and
                          images {list: num) -> dict{'subset', 'class_name', 'filepath'} into npy file

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        memory = []
        print('Indexing {}...{}...'.format(target, subset))

        # Quick first pass to find total for tqdm bar
        subset_len = 0

        # determine target's path
        if subset == 'background':
            folder_name = 'train'
        elif subset == 'evaluation':
            folder_name = 'val'
        else:
            folder_name = 'test'
        if target in ('CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi'):
            target_path_root = DATA_PATH + '/meta-dataset/{}'.format(target)
        elif target in ('clipart_84', 'infograph_84', 'painting_84', 'quickdraw_84', 'real_84', 'sketch_84'):
            target_path_root = DATA_PATH + '/DomainNet/{}'.format(target)
        else:
            target_path_root = DATA_PATH + '/{}'.format(target)

        if preload:
            if os.path.exists(target_path_root + '/{}_images_memory.npy'.format(folder_name)):
                print('{}: load {}/{}_images_memory.npy'.format(Meta, target_path_root, folder_name))
                data = torch.load(target_path_root + '/{}_images_memory.npy'.format(folder_name))
                images = data['images']
                memory = data['memory']
                return images, memory
            else:
                print('{}: load images into memory.'.format(Meta))

        target_path = target_path_root + '/{}'.format(folder_name)
        print('{}: construct npy from target_path: {}'.format(Meta, target_path))

        for root, folders, files in os.walk(target_path):
            subset_len += len([f for f in files if f.endswith('.jpg') or f.endswith('.JPG')])
        if subset_len == 0:
            raise Exception('image file not ended with jpg.')
        print('find {} images.'.format(subset_len))

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(target_path):
            if len(files) == 0:
                continue

            class_name = root.split(os.sep)[-1]     # linux / ; windows \\
            # 014.Indigo_Bunting

            for f in [f for f in files if f.endswith('.jpg') or f.endswith('.JPG')]:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })
                # filepath: //10.20.2.245/datasets/datasets/meta-dataset/CUB_Bird/val
                #               \\014.Indigo_Bunting\\Indigo_Bunting_0001_12469.jpg

                # load memory
                if preload:
                    instance = Image.open(os.path.join(root, f))     # JpegImageFile, 84x84
                    instance = transform(instance)                   # [3, 84, 84]
                    memory.append(instance)

        progress_bar.close()

        # store npy
        if preload:
            memory = torch.stack(memory)
            print('{}: store {}/{}_images_memory.npy'.format(Meta, target_path_root, folder_name))
            state = {'images': images, 'memory': memory}
            torch.save(state, target_path_root + '/{}_images_memory.npy'.format(folder_name))

        return images, memory


class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)


class MultiDataset(Dataset):
    def __init__(self, dataset_list: List[Dataset]):
        """Dataset class representing a list of datasets

        # Arguments:
            :param dataset_list: need to first prepare each sub-dataset
        """
        self.dataset_list = dataset_list
        self.datasetid_to_class_id = self.label_mapping()

        # cat all df in dataset_list
        # e.g., CUB Bird
        #   class_name            filepath    subset    id   class_id         {Bird: 960}
        # 0  014.Indigo_Bunting              ...         0          0
        # 1  014.Indigo_Bunting              ...         1          0
        # 2  014.Indigo_Bunting              ...         2          0
        # 3  014.Indigo_Bunting              ...         3          0
        # 4  014.Indigo_Bunting              ...         4          0
        # store origin dataset_id into column[origin_dataset_id] for each df
        for idx, dataset in enumerate(dataset_list):
            dataset.df['origin_dataset_id'] = idx
        self.df = pd.concat([dataset.df for dataset in dataset_list], keys=[dataset.target for dataset in dataset_list])
        # store origin id into column[origin_id]
        self.df.rename(columns={'id': 'origin_id'}, inplace=True)
        # store origin class_id into column[origin_class_id]
        self.df.rename(columns={'class_id': 'origin_class_id'}, inplace=True)
        # update id with offset
        self.df['id'] = range(len(self.df))
        # update class_id with datasetid_to_class_id
        self.df = self.df.assign(class_id=self.df['id'].apply(lambda c: self.datasetid_to_class_id[c]))
        #               class_name   ...    origin_id   origin_class_id origin_dataset_id   id  class_id
        # CIFAR100_84/0       crab   ...            0                 0                 0    0         0
        #            /1       crab   ...            1                 0                 0    1         0
        #            /2       crab   ...            2                 0                 0    2         0
        #            /3       crab   ...            3                 0                 0    3         0
        #            /4       crab   ...            4                 0                 0    4         0
        # ...
        # CIFAR10_84 /0       bird   ...            0                 0                 1 1000       100
        #            /1       bird   ...            1                 0                 2 1001       100

        self.datasetid_to_origin_id = list(self.df['origin_id'])
        self.datasetid_to_origin_dataset_id = list(self.df['origin_dataset_id'])

    def label_mapping(self) -> Dict:
        """
        generate mapping dict from datasetid to global class id.
        :return: datasetid_to_class_id
        """
        datasetid_to_class_id = dict()
        index_offset = 0
        class_id_offset = 0
        for dataset in self.dataset_list:
            datasetid_to_class_id.update(
                dict(zip(map(lambda id:             id + index_offset,    dataset.datasetid_to_class_id.keys()),
                         map(lambda class_id: class_id + class_id_offset, dataset.datasetid_to_class_id.values())))
            )

            index_offset = index_offset + len(dataset)
            class_id_offset = class_id_offset + dataset.num_classes()

        return datasetid_to_class_id

    def __getitem__(self, item):
        # dataset_id, index = self.index_mapping(item)
        dataset_id, index = self.datasetid_to_origin_dataset_id[item], self.datasetid_to_origin_id[item]
        instance, true_label = self.dataset_list[dataset_id][index]     # true_label is the label(int) in sub-dataset
        label = self.datasetid_to_class_id[item]                        # label is the label(int) with offset
        return instance, label

    def __len__(self):
        return sum([len(dataset) for dataset in self.dataset_list])

    def num_classes(self):
        sum([dataset.num_classes() for dataset in self.dataset_list])

    def index_mapping(self, index) -> (int, int):
        """
        A mapping method to map index (in __getitem__ method) to the index in the corresponding dataset.

        :param index:
        :return: dataset_id, item
        """
        index_origin = index
        for dataset_id, dataset in enumerate(self.dataset_list):
            if index < len(dataset):
                return dataset_id, index
            else:
                index = index - len(dataset)

        raise(ValueError, f'index exceeds total number of instances, index {index_origin}')


if __name__ == "__main__":
    # debug on MultiDataset
    evaluation = MultiDataset([Meta('evaluation', 'CUB_Bird', preload=True),
                               Meta('evaluation', 'FGVC_Aircraft', preload=True)])

    print(evaluation[1000][0].shape, evaluation[1000][1])
    # Indexing CUB_Bird...evaluation...
    # 1220it [00:00, 19339.69it/s]
    # Indexing DTD_Texture...evaluation...
    # 1209it [00:00, 22444.32it/s]
    # Indexing FGVC_Aircraft...evaluation...
    # 2020it [00:00, 23293.17it/s]
    # torch.Size([3, 84, 84]) 15

    # print(evaluation.df)
    from matplotlib import pyplot as plt
    plt.imshow(np.transpose(evaluation[0][0].numpy(), (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(evaluation[1000][0].numpy(), (1, 2, 0)))
    plt.show()
