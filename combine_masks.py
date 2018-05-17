import cv2
import numpy as np
from glob import glob


def get_bool_mask(path):
	img = cv2.imread(path,0)
	img = cv2.resize(img,target_size)
	
	mask = np.zeros(target_size)
	mask[img>127] = 1
	return mask

left_paths = glob('data/all/leftMask/*.png')
right_paths = glob('data/all/rightMask/*.png')
target_size = (256,256)

for left_path,right_path in zip(left_paths,right_paths):
	
	name = left_path.split('/')[-1]	

	left_mask = get_bool_mask(left_path).astype(bool)
	right_mask = get_bool_mask(right_path).astype(bool)
	combined_mask = left_mask + right_mask

	print name
	print combined_mask.shape
	cv2.imwrite('data/all/combined_mask/'+name,combined_mask*255)



