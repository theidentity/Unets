import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from glob import glob
import shutil
import os


def get_train_test_names():

	img_paths = glob('data/all/images_mtgm/*.png')
	names = [x.split('/')[-1] for x in img_paths]
	print len(names)

	X = names
	train_X,test_X = train_test_split(X,test_size=0.2,random_state=42)
	return train_X,test_X

def create_folder(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)

def create_folder_skeleton(base_path):
	skeleton = ['train/images/input','train/masks/input','validation/images/input','validation/masks/input']
	for item in skeleton:
		create_folder(base_path+item)

def copy_images(in_paths,out_paths):
	for src,dst in zip(in_paths,out_paths):
		shutil.copy(src,dst)

def copy_sets(train,test,base_path):

	img_in_paths = [''.join(['data/all/images_mtgm/',name]) for name in train]
	img_out_paths = [''.join([base_path,'train/images/input/',name]) for name in train]
	copy_images(img_in_paths,img_out_paths)

	mask_in_paths = [''.join(['data/all/combined_mask/',name]) for name in train]
	mask_out_paths = [''.join([base_path,'train/masks/input/',name]) for name in train]
	copy_images(mask_in_paths,mask_out_paths)

	img_in_paths = [''.join(['data/all/images_mtgm/',name]) for name in test]
	img_out_paths = [''.join([base_path,'validation/images/input/',name]) for name in test]
	copy_images(img_in_paths,img_out_paths)

	mask_in_paths = [''.join(['data/all/combined_mask/',name]) for name in test]
	mask_out_paths = [''.join([base_path,'validation/masks/input/',name]) for name in test]
	copy_images(mask_in_paths,mask_out_paths)

def get_stats(base_path):
	for root, directories,filenames in os.walk(base_path):
		if len(filenames)>0:
			print root,":",len(filenames)

if __name__ == '__main__':
	# create_folder_skeleton(base_path = 'data/foldA/')
	# train,test = get_train_test_names()
	# copy_sets(train,test,base_path='data/foldA/')
	get_stats(base_path='data/foldA')
	get_stats(base_path='data/test/')
