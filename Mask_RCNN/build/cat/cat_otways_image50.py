"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
# ROOT_DIR = "drive/Mask_RCNN_cobbob/"
ROOT_DIR = "./"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
imagenet_WEIGHTS_PATH = os.path.join(ROOT_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CatConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cat"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    BACKBONE = "resnet50"

    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Number of classes (including background)
    NUM_CLASSES = 1 + 43  # Background + cat1 + cat2 + not_defined

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7



############################################################
#  Dataset
############################################################

class CatDataset(utils.Dataset):

    def load_cat(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cat", 1, "otway_tabby_vincent")
        self.add_class("cat", 2, "otway_tabby_stumpy")
        self.add_class("cat", 3, "otway_tabby_spatchcock")
        self.add_class("cat", 4, "otway_tabby_scoot")
        self.add_class("cat", 5, "otway_tabby_roswell")
        self.add_class("cat", 6, "otway_tabby_pauline")
        self.add_class("cat", 7, "otway_tabby_patrick")
        self.add_class("cat", 8, "otway_tabby_patricia")
        self.add_class("cat", 9, "otway_tabby_meg")
        self.add_class("cat", 10, "otway_tabby_jesus")
        self.add_class("cat", 11, "otway_tabby_javier")
        self.add_class("cat", 12, "otway_tabby_howard")
        self.add_class("cat", 13, "otway_tabby_hank")
        self.add_class("cat", 14, "otway_tabby_crass")
        self.add_class("cat", 15, "otway_tabby_charlotte")
        self.add_class("cat", 16, "otway_tabby_celeste")
        self.add_class("cat", 17, "otway_tabby_catsup")
        self.add_class("cat", 18, "otway_tabby_bennett")
        self.add_class("cat", 19, "otway_swirl_venus")
        self.add_class("cat", 20, "otway_swirl_tamsyn")
        self.add_class("cat", 21, "otway_swirl_stimmy")
        self.add_class("cat", 22, "otway_swirl_stewart")
        self.add_class("cat", 23, "otway_swirl_spicy")
        self.add_class("cat", 24, "otway_swirl_skip")
        self.add_class("cat", 25, "otway_swirl_sirloin")
        self.add_class("cat", 26, "otway_swirl_peter")
        self.add_class("cat", 27, "otway_swirl_penelope")
        self.add_class("cat", 28, "otway_swirl_pav")
        self.add_class("cat", 29, "otway_swirl_new_")
        self.add_class("cat", 30, "otway_swirl_logan")
        self.add_class("cat", 31, "otway_swirl_knuckles")
        self.add_class("cat", 32, "otway_swirl_kathleen")
        self.add_class("cat", 33, "otway_swirl_jim")
        self.add_class("cat", 34, "otway_swirl_jill")
        self.add_class("cat", 35, "otway_swirl_golding")
        self.add_class("cat", 36, "otway_swirl_edgy")
        self.add_class("cat", 37, "otway_swirl_clinton")
        self.add_class("cat", 38, "otway_swirl_chowder")
        self.add_class("cat", 39, "otway_swirl_bluey")
        self.add_class("cat", 40, "otway_swirl_angelica")
        self.add_class("cat", 41, "otway_ginger_holyspirit")
        self.add_class("cat", 42, "otway_black_kingfluffy")
        self.add_class("cat", 43, "other")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            
#             if type(a['regions']) is dict:
#                 polygons = [r['shape_attributes'] for r in a['regions'].values()]
#             else:
#                 polygons = [r['shape_attributes'] for r in a['regions']] 
            
            rects = [r['shape_attributes'] for r in a['regions']] 
            name = [r['region_attributes']['type'] for r in a['regions']]
            name_dict = {"otway_tabby_vincent":1, "otway_tabby_stumpy":2, "otway_tabby_spatchcock":3, "otway_tabby_scoot":4,
                         "otway_tabby_roswell":5, "otway_tabby_pauline":6, "otway_tabby_patrick":7, "otway_tabby_patricia":8,
                         "otway_tabby_meg":9,"otway_tabby_jesus":10,"otway_tabby_javier":11,"otway_tabby_howard":12,"otway_tabby_hank":13,
                         "otway_tabby_crass":14,"otway_tabby_charlotte":15,"otway_tabby_celeste":16,"otway_tabby_catsup":17,"otway_tabby_bennett":18
                         ,"otway_swirl_venus":19,"otway_swirl_tamsyn":20,"otway_swirl_stimmy":21,"otway_swirl_stewart":22,"otway_swirl_spicy":23
                         ,"otway_swirl_skip":24,"otway_swirl_sirloin":25,"otway_swirl_peter":26,"otway_swirl_penelope":27,"otway_swirl_pav":28
                         ,"otway_swirl_new_":29,"otway_swirl_logan":30,"otway_swirl_knuckles":31,"otway_swirl_kathleen":32,"otway_swirl_jim":33
                         ,"otway_swirl_jill":34,"otway_swirl_golding":35,"otway_swirl_edgy":36,"otway_swirl_clinton":37,"otway_swirl_chowder":38
                         ,"otway_swirl_bluey":39,"otway_swirl_angelica":40,"otway_ginger_holyspirit":41,"otway_black_kingfluffy":42,"other":43}

            name_id = [name_dict[a] for a in name]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            # 注意可以统一图片大小就不需要把image导入了
            image_path = os.path.join(dataset_dir, a['filename'])
            # image = skimage.io.imread(image_path)
            # height, width = image.shape[:2]
            height = a['height']
            width = a['width']

            self.add_image(
                "cat",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                class_id=name_id,
                width=width, height=height,
                polygons=rects)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cat":
            return super(self.__class__, self).load_mask(image_id)

        name_id = image_info["class_id"]
        print(name_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr, cc = skimage.draw.rectangle((p['y'], p['x']), extent=(p['height'], p['width']))
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        
        class_ids = np.array(name_id, dtype=np.int32)
        
        return (mask.astype(np.bool), class_ids)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cat":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CatDataset()
    dataset_train.load_cat(args.dataset, "train")
    dataset_train.prepare()


    # Validation dataset
    dataset_val = CatDataset()
    dataset_val.load_cat(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect cats.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cat/dataset/",
                        help='Directory of the cat dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--image', required=False,
    #                     metavar="path or URL to image",
    #                     help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    args = parser.parse_args()


    # Validate arguments
    # if args.command == "train":
    #     assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "splash":
    #     assert args.image or args.video,\
    #            "Provide --image or --video to apply color splash"
    assert args.dataset, "Argument --dataset is required for training"
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    # if args.command == "train":
    #     config = CatConfig()
    # else:
    #     class InferenceConfig(CatConfig):
    #         # Set batch size to 1 since we'll be running inference on
    #         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #         GPU_COUNT = 1
    #         IMAGES_PER_GPU = 1
    #     config = InferenceConfig()
    config = CatConfig()
    config.display()

    # Create model
    # if args.command == "train":
    #     model = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=args.logs)
    # else:
    #     model = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs)
    model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = imagenet_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    # if args.command == "train":
    #     train(model)
    # elif args.command == "splash":
    #     detect_and_color_splash(model, image_path=args.image,
    #                             video_path=args.video)
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'splash'".format(args.command))
    train(model)