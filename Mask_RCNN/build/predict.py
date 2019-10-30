#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.




import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# ROOT_DIR = "drive/Mask_RCNN/"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/cat/"))  # To find local version
import cat_annya_imagenet50 as cat

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
CAT_MODEL_PATH = os.path.join(ROOT_DIR, "logs/annya_imagenet50_0040.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(CAT_MODEL_PATH):
    utils.download_trained_weights(CAT_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "samples/cat/annya_dataset/val")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.




class InferenceConfig(cat.CatConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    # NUM_CLASSES = 1 + 4
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(CAT_MODEL_PATH, by_name=True)


# Run Object Detection


# Load a random image from the images folder
filename = os.path.join(IMAGE_DIR,'annya_swirl_ben-184.JPG')
image = skimage.io.imread(filename)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
print(r)
visualize.display_instances(image=image, boxes=r['rois'],masks=r['masks'], class_ids=r['class_ids'],
                            class_names=class_names, scores=r['scores'],show_mask=False)


#load dataset
dataset_dir = os.path.join(ROOT_DIR, "samples/cat/Feral_cats/annya_dataset")
dataset = cat.CatDataset()
dataset.load_cat(dataset_dir,"val")
dataset.prepare()
class_names = dataset.class_names

# calculate mAP
class_APs = {}
for i in range(1, len(class_names)):
    class_APs.setdefault(i, [])

for image_id in dataset.image_ids:
    # Load image
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config,
                               image_id, use_mini_mask=False)
    # Run object detection
    results = model.detect([image], verbose=0)
    # Compute AP
    r = results[0]
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r['rois'], r['class_ids'], r['scores'], r['masks'])

    info = dataset.image_info[image_id]
    print(info["id"], "AP", AP, "precisions", precisions, "recalls", recalls, "class", r['class_ids'], "scores",
          r['scores'], "rois", r['rois'])

    class_APs[gt_class_id[0]].append(AP)

APs = []
for key in class_APs.keys():
    if class_APs[key] != []:
        tmean = np.mean(class_APs[key])
    else:
        tmean = 0
    APs.append(tmean)
    print(class_names[key], "mAP @ IoU=50: ", tmean)

print("mAP @ IoU=50: ", np.mean(APs))


