"""
Run this script to prepare 4 meta-dataset dataset: CUB_Bird, DTD_Texture, FGVC_Aircraft, FGVCx_Fungi.

We follow the dataset settings as in HSML: https://arxiv.org/abs/1905.05301.

1. Download multidataset.zip from https://drive.google.com/file/d/1IJk93N48X0rSL69nQ1Wr-49o8u0e75HM/view and place in
    DATA_PATH/
2. extract multidataset.zip and obtain meta-dataset folder containing 4 sub-folders:
    CUB_Bird, DTD_Texture, FGVC_Aircraft, FGVCx_Fungi.
"""

# """
# Run this script to prepare the bird dataset.
#
# This script resize images into shape (84, 84).
#
# Raw Birds folder structure:
# bird \
#     |-- images \
#         |-- class name \
#             |-- imgs \
# totally 200 classes with different number of images for each.
#
# 1. Download images.tgz from http://www.vision.caltech.edu/visipedia/CUB-200.html and place in
#     DATA_PATH/bird
# 2. extract images.tgz and images.tar and obtain images folder.
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
# rmdir(DATA_PATH + '/bird/images_background')
# rmdir(DATA_PATH + '/bird/images_evaluation')
# mkdir(DATA_PATH + '/bird/images_background')
# mkdir(DATA_PATH + '/bird/images_evaluation')
#
# # Find class identities
# classes = []
# for file in os.listdir(DATA_PATH + '/bird/images/'):
#     if os.path.isdir(DATA_PATH + '/bird/images/' + file):
#         classes.append(file)
#
# assert len(classes) == 200
#
# # Train/test split
# np.random.seed(0)
# np.random.shuffle(classes)
# background_classes, evaluation_classes = classes[:160], classes[160:]
#
# # Create class folders
# for c in background_classes:
#     mkdir(DATA_PATH + f'/bird/images_background/{c}/')
#
# for c in evaluation_classes:
#     mkdir(DATA_PATH + f'/bird/images_evaluation/{c}/')
#
# # Move images to correct location
# for root, _, files in os.walk(DATA_PATH + '/bird/images'):
#     print(root)
#     for f in tqdm(files):
#         if f.endswith('.jpg') and not f.startswith('._'):
#             # resize
#             img = Image.open(root + '/' + f)
#             img = img.resize((84, 84))
#             class_name = root.split(os.sep)[-1]
#             image_name = f
#             # Send to correct folder
#             subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
#             dst = DATA_PATH + f'/bird/{subset_folder}/{class_name}/{image_name}'
#
#             # src = f'{root}/{f}'
#             img.save(dst)
#             # shutil.copy(src, dst)
