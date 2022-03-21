"""
Run this script to prepare 4 meta-dataset dataset: CUB_Bird, DTD_Texture, FGVC_Aircraft, FGVCx_Fungi.

We follow the dataset settings as in HSML: https://arxiv.org/abs/1905.05301.

1. Download multidataset.zip from https://drive.google.com/file/d/1IJk93N48X0rSL69nQ1Wr-49o8u0e75HM/view and place in
    DATA_PATH/
2. extract multidataset.zip and obtain meta-dataset folder containing 4 sub-folders:
    CUB_Bird, DTD_Texture, FGVC_Aircraft, FGVCx_Fungi.
"""

# """
# Run this script to prepare the aircraft dataset.
#
# This script resize images into shape (84, 84).
#
# Raw aircraft folder structure:
# aircraft \
#     |-- fgvc-aircraft-2013b \
#         | -- data \
#             | -- images \
#                 |-- imgs \
#             |-- images_variant_test.txt \
#             |-- images_variant_train.txt \
#             |-- images_variant_val.txt \
# three txt files has a structure for each row: "img_name class_name".
#     replace space character in the class_name with '-'. e.g., 'Beechcraft 1900'
#     replace '/' character in the class_name with '-'.   e.g., 'F/A-18'
# Dataset contains 100 classes and 100 image instances each class.
#
# 1. Download fgvc-aircraft-2013b.tar.gz from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ and place in
#     DATA_PATH/aircraft
# 2. extract fgvc-aircraft-2013b.tar.gz and fgvc-aircraft-2013b.tar and obtain fgvc-aircraft-2013b folder.
# 3. Run the script
# """
# from tqdm import tqdm as tqdm
# import numpy as np
# # import shutil
# from PIL import Image
# import os
#
# from config import DATA_PATH
# from few_shot.utils import mkdir, rmdir
#
#
# # Clean up folders
# rmdir(DATA_PATH + '/aircraft/images_background')
# rmdir(DATA_PATH + '/aircraft/images_evaluation')
# mkdir(DATA_PATH + '/aircraft/images_background')
# mkdir(DATA_PATH + '/aircraft/images_evaluation')
#
# # collect img names and labels
# img_data = {}
# with open(DATA_PATH + '/aircraft/fgvc-aircraft-2013b/data/images_variant_test.txt', 'r') as f:
#     for line in f.readlines():
#         line = line.strip('\n').replace('/', '-')       # avoid 'F/A-18' case
#         line = line.split(' ')
#         img_name = line[0]
#         label = '-'.join(line[1:])
#         if label in img_data.keys():
#             img_data[label].append(img_name)
#         else:
#             img_data[label] = [img_name]
# with open(DATA_PATH + '/aircraft/fgvc-aircraft-2013b/data/images_variant_train.txt', 'r') as f:
#     for line in f.readlines():
#         line = line.strip('\n').replace('/', '-')       # avoid 'F/A-18' case
#         line = line.split(' ')
#         img_name = line[0]
#         label = '-'.join(line[1:])
#         if label in img_data.keys():
#             img_data[label].append(img_name)
#         else:
#             img_data[label] = [img_name]
# with open(DATA_PATH + '/aircraft/fgvc-aircraft-2013b/data/images_variant_val.txt', 'r') as f:
#     for line in f.readlines():
#         line = line.strip('\n').replace('/', '-')       # avoid 'F/A-18' case
#         line = line.split(' ')
#         img_name = line[0]
#         label = '-'.join(line[1:])
#         if label in img_data.keys():
#             img_data[label].append(img_name)
#         else:
#             img_data[label] = [img_name]
# print('least and most numbers of image instances:',
#       np.min([len(imgs) for imgs in img_data.values()]),
#       np.max([len(imgs) for imgs in img_data.values()]))
# print('number of classes:', len(img_data.keys()))
#
# # Find class identities
# classes = list(img_data.keys())
# assert len(classes) == 100
# print(classes)
#
# # Train/test split
# np.random.seed(0)
# np.random.shuffle(classes)
# background_classes, evaluation_classes = classes[:80], classes[80:]
#
# # Create class folders
# for c in background_classes:
#     mkdir(DATA_PATH + f'/aircraft/images_background/{c}/')
#
# for c in evaluation_classes:
#     mkdir(DATA_PATH + f'/aircraft/images_evaluation/{c}/')
#
# # Move images to correct location
# for class_name in classes:
#     img_list = img_data[class_name]
#     for image_name in tqdm(img_list):
#         img = Image.open(DATA_PATH + f'/aircraft/fgvc-aircraft-2013b/data/images/{image_name}.jpg')
#         img = img.resize((84, 84))
#         subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
#         dst = DATA_PATH + f'/aircraft/{subset_folder}/{class_name}/{image_name}.jpg'
#         img.save(dst)
