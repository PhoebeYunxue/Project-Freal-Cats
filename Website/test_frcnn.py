from __future__ import division
import cv2
import numpy as np
import pickle

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

def define_C(model_name):
	# if model_name == "G":
	# 	config_output_filename = './configs/config_glenelg.pickle'
	if model_name == "A":
		config_output_filename = './configs/config_annya.pickle'
	elif model_name == "C":
		config_output_filename = './configs/config_cobbob.pickle'
	elif model_name == "H":
		config_output_filename = './configs/config_hotspur.pickle'
	elif model_name == "M":
		config_output_filename = './configs/config_mt.pickle'
	elif model_name == "O":
		config_output_filename = './configs/config_otways.pickle'

	with open(config_output_filename, 'rb') as f_in:
		C = pickle.load(f_in)

	# turn off any data augmentation at test time
	# if model_name == "G":
	# 	C.model_path = './models/model_frcnn_resnet_glenelg.hdf5'
	if model_name == "A":
		C.model_path = './models/model_frcnn_resnet_annya.hdf5'
	elif model_name == "C":
		C.model_path = './models/model_frcnn_resnet_cobbob.hdf5'
	elif model_name == "H":
		C.model_path = './models/model_frcnn_resnet_hotspur.hdf5'
	elif model_name == "M":
		C.model_path = './models/model_frcnn_resnet_mt_clay.hdf5'
	elif model_name == "O":
		C.model_path = './models/model_frcnn_resnet_otways.hdf5'

	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False
	C.num_rois = 32

	return C

def get_new_img_size(width, height, img_min_side=300):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height