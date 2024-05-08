import torch
import os
import random
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import zipfile
import yaml
import PIL

from IPython.display import Image  # for displaying images
from sklearn.model_selection import train_test_split
from pylabel import importer
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw

logging.getLogger().setLevel(logging.CRITICAL)
random.seed(42)

# Specify path to the coco.json file
path_to_annotations = r"./coco_instances.json"
# Specify the path to the images (if they are in a different folder than the annotations)
path_to_images = r"./images_raw"

# Import the dataset into the pylable schema
dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="BCCD_coco")
dataset.df.head(5)

print(f"Number of images: {dataset.analyze.num_images}")
print(f"Number of classes: {dataset.analyze.num_classes}")
print(f"Classes:{dataset.analyze.classes}")
print(f"Class counts:\n{dataset.analyze.class_counts}")
print(f"Path to annotations:\n{dataset.path_to_annotations}")

# This cell may take some time depending on the size of the dataset.
dataset.path_to_annotations = "labels"
dataset.export.ExportToYoloV5(output_path='text_files')

"remove images with no annotations"
for anno in os.listdir('./text_files'):
    if anno.split('.')[0] + '.jpg' not in os.listdir('./images_raw'):
        os.remove('./text_files/' + anno)

# Read images and annotations
image_dir = r'./images_raw'
images = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
annotations = [os.path.join('./text_files', x) for x in os.listdir('./text_files') if x[-3:] == "txt"]

images.sort()
annotations.sort()
#
# # Split the dataset into train-valid-test splits
# train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.2,
#                                                                                 random_state=1)
# val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,
#                                                                               test_size=0.5, random_state=1)

# print(len(train_images), " ", len(train_annotations))

# #Utility function to move images
# def move_files_to_folder(list_of_files, destination_folder):
#     for f in list_of_files:
#         try:
#             shutil.move(f, destination_folder)
#         except:
#             print(f)
#             assert False
#
# # Move the splits into their folders
# move_files_to_folder(train_images, 'images/train')
# move_files_to_folder(val_images, 'images/val/')
# move_files_to_folder(test_images, 'images/test/')
# move_files_to_folder(train_annotations, 'annotations/train/')
# move_files_to_folder(val_annotations, 'annotations/val/')
# move_files_to_folder(test_annotations, 'annotations/test/')

yaml_params = {}
with open(r'dataset.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    yaml_file_list = yaml.load(file, Loader=yaml.FullLoader)
    yaml_params = yaml_file_list
    print(yaml_file_list)

# Adjusting the parameters of the yaml file
yaml_params['path'] = 'images'
yaml_params['train'] = 'train'
yaml_params['val'] = 'val'
yaml_params['test'] = 'test'
print(yaml_params)



# Moving the dataset.yaml inside the yolov5/data folder.
shutil.move("dataset.yaml", "yolov5/data")

shutil.move("./test_image.jpg", "./yolov5")
