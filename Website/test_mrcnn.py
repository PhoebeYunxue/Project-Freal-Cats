#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
#
# Import Mask RCNN
from mrcnn.config import Config
import mrcnn.model as modellib

# ## Configurations.

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    def __init__(self,num_class):
        Config.__init__(self,num_class)

def loadmodel(model_name):
    MODEL_DIR = "./models"
    if model_name == "A":
        NUM_CLASSES = 1 + 8
        CAT_MODEL_PATH = "./models/annya_imagenet50_0040.h5"
    elif model_name == "C":
        NUM_CLASSES = 1 + 12
        CAT_MODEL_PATH = "./models/cobbob_imagenet50_0040.h5"
    elif model_name == "H":
        NUM_CLASSES = 1 + 7
        CAT_MODEL_PATH = "./models/hotspur_imagenet50_0040.h5"
    elif model_name == "M":
        NUM_CLASSES = 1 + 10
        CAT_MODEL_PATH = "./models/mtclay_imagenet50_0040.h5"
    elif model_name == "O":
        NUM_CLASSES = 1 + 43
        CAT_MODEL_PATH = "./models/otways_imagenet50_0040.h5"

    config = InferenceConfig(NUM_CLASSES)
    # Create Model and Load Trained Weights
    # Local path to trained weights file
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained
    model.load_weights(CAT_MODEL_PATH, by_name=True)

    return model

def defineClass(model_name):
    if model_name == "A":
        class_names = ["BG", "annya_swirl_ben", "annya_swirl_gilbert", "annya_swirl_lola", "annya_swirl_motley",
                       "annya_tabby_danny", "annya_tabby_joe", "annya_tabby_stumps", "other"]
    elif model_name == "C":
        class_names = ["BG", "c_black_mangy", "c_swirl_eskimo", "c_swirl_pattsy", "c_swirl_sprout",
                         "c_swirl_sterling", "c_swirl_willy", "c_tabby_brisket", "c_tabby_charlie",
                         "c_tabby_quasimodo","c_tabby_shelby","c_tabby_sleuce","other",]
    elif model_name == "H":
        class_names = ["BG", "hotspur_swirl_1", "hotspur_swirl_2", "hotspur_swirl_3", "hotspur_tabby_1",
                         "hotspur_tabby_2", "hotspur_tabby_3", "other"]
    elif model_name == "M":
        class_names = ["BG", "m_swirl_arnold", "m_swirl_ghost", "m_swirl_lynxy", "m_swirl_sheila",
                       "m_swirl_socksy","m_swirl_spruce", "m_swirl_stubby","m_tabby_cassidy","m_tabby_murray","other"]
    elif model_name == "O":
        class_names = ["BG", "otway_tabby_vincent", "otway_tabby_stumpy", "otway_tabby_spatchcock",
                       "otway_tabby_scoot", "otway_tabby_roswell", "otway_tabby_pauline", "otway_tabby_patrick", "otway_tabby_patricia",
                       "otway_tabby_meg", "otway_tabby_jesus", "otway_tabby_javier", "otway_tabby_howard", "otway_tabby_hank",
                       "otway_tabby_crass", "otway_tabby_charlotte", "otway_tabby_celeste", "otway_tabby_catsup", "otway_tabby_bennett",
                       "otway_swirl_venus", "otway_swirl_tamsyn", "otway_swirl_stimmy", "otway_swirl_stewart", "otway_swirl_spicy",
                       "otway_swirl_skip", "otway_swirl_sirloin", "otway_swirl_peter", "otway_swirl_penelope", "otway_swirl_pav",
                       "otway_swirl_new_", "otway_swirl_logan", "otway_swirl_knuckles", "otway_swirl_kathleen", "otway_swirl_jim",
                       "otway_swirl_jill", "otway_swirl_golding", "otway_swirl_edgy", "otway_swirl_clinton", "otway_swirl_chowder",
                       "otway_swirl_bluey", "otway_swirl_angelica", "otway_ginger_holyspirit", "otway_black_kingfluffy", "other"]

    return class_names




