import numpy as np

from keras.layers import Conv2D,MaxPool2D,UpSampling2D
from keras.layers import Input,Dropout,merge,concatenate
from keras.models import Model,save_model,load_model
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from keras.optimizers import Adam,SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator

import datetime
import itertools
import cv2
from glob import glob


class Unet(object):
	"""docstring for Unet"""
	def __init__(self):

		self.img_rows,self.img_cols = 256,256
		self.batch_size = 4
		self.seed = 42

		self.train_img_path = 'data/foldA/train/images'
		self.train_mask_path = 'data/foldA/train/masks'
		self.validation_img_path = 'data/foldA/validation/images'
		self.validation_mask_path = 'data/foldA/validation/masks'
		self.test_img_path = 'data/test'

		self.name = 'Unet'
		self.save_path = ''.join(['models/',self.name,'_best','.h5'])

		self.model = self.get_model()

	def get_model(self):

		inputs = Input((self.img_rows, self.img_cols,1))
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = concatenate([drop4,up6], axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = concatenate([conv3,up7], axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = concatenate([conv2,up8], axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = concatenate([conv1,up9], axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

		model = Model(inputs,conv10)
		return model

	def build_model(self,lr=1e-4):
		
		opt = Adam(lr=lr)
		self.model.compile(
			optimizer = opt,
			loss = 'binary_crossentropy',
			metrics = ['accuracy']
			)

	def train_gen(self,path):
		img_gen = ImageDataGenerator(
			zoom_range = 0.2,
			width_shift_range = 0.2,
			height_shift_range = 0.2,
			horizontal_flip = False,
			rotation_range = 10.0,
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed = self.seed,
			class_mode = None,
			color_mode = 'grayscale'
			)

		return img_gen

	def test_gen(self,path):
		img_gen = ImageDataGenerator(
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed =self.seed,
			class_mode = None,
			color_mode = 'grayscale'
			)

		return img_gen

	def get_train_generator(self):
		img_gen = self.train_gen(self.train_img_path)
		mask_gen = self.train_gen(self.train_mask_path)
		return itertools.izip(img_gen,mask_gen)

	def get_validation_generator(self):
		img_gen = self.test_gen(self.validation_img_path)
		mask_gen = self.test_gen(self.validation_mask_path)
		return itertools.izip(img_gen,mask_gen)

	def get_test_generator(self,path):
		img_gen = self.test_gen(path)
		return img_gen

	def get_callbacks(self):
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=self.save_path, verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		return [early_stopping,checkpointer,tensorboard]

	def train(self,lr=1e-4,num_epochs=1):

		self.build_model(lr)

		train_generator = self.get_train_generator()
		validation_generator = self.get_validation_generator()

		hist = self.model.fit_generator(
			generator = train_generator,
			epochs  = num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			verbose = 1,
			steps_per_epoch = 110//self.batch_size + 1,
			validation_steps = 28//self.batch_size + 1
			)

	def continue_training(self,lr=1e-4,num_epochs=1):

		self.model = load_model(self.save_path)
		self.build_model(lr)

		train_generator = self.get_train_generator()
		validation_generator = self.get_validation_generator()

		hist = self.model.fit_generator(
			generator = train_generator,
			epochs  = num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			verbose = 1,
			steps_per_epoch = 110//self.batch_size + 1,
			validation_steps = 28//self.batch_size + 1
			)

	def get_outputs(self):

		model = load_model(self.save_path)
		test_generator = self.get_test_generator(self.test_img_path)


		for i in xrange(20//4):

			X = test_generator.next()
			y = model.predict(X)

			X = self.normalize_array(X,0,255)
			y = self.normalize_array(y,0,255)

			for j,(img,mask) in enumerate(zip(X,y)):
				print i,j
				print img.shape
				print mask.shape

				name = ''.join(['tmp/',str(i),'_',str(j),'.jpg'])
				out = np.hstack([img,mask])
				cv2.imwrite(name,out)


	def test_model(self):
		
		model = load_model(self.save_path)
		test_generator = self.get_test_generator(self.test_img_path)


		for i in xrange(663//4):

			print (663//4)-i,'remaining'

			X = test_generator.next()
			y = model.predict(X)

			X = self.normalize_array(X,0,255)
			y = self.normalize_array(y,0,255)

			left = np.vstack([x for x in X])
			right = np.vstack([x for x in y])
			out = np.hstack([left,right])

			name = ''.join(['tmp/wall/',str(i),'.jpg'])
			cv2.imwrite(name,out)
			
	def generate_masks(self):
		model = load_model(self.save_path)
		names = [x.split('/')[-1] for x in glob('data/CXR_png/*.png')]

		for name in names:
			print name 

			in_path = ''.join(['data/CXR_png/',name])
			out_path_mask = ''.join(['tmp/seperate/masks/',name])
			out_path_img = ''.join(['tmp/seperate/cropped_images/',name])

			img = cv2.imread(in_path,0)
			img = cv2.resize(img,(self.img_rows,self.img_cols))
			img = img/255.0
			img = img.reshape((-1,self.img_rows,self.img_cols,1))

			y = model.predict(img)[0]
			y = self.normalize_array(y,0,255)
			print y.shape
			cv2.imwrite(out_path_mask,y)

			crop_img = img[0].copy()
			crop_img = self.normalize_array(crop_img,0,255)
			crop_img[y<127] = 0
			print crop_img.shape
			cv2.imwrite(out_path_img,crop_img)

	def normalize_array(self,arr,lower=0,upper=255):
	    arr = (upper-lower)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
	    return arr


if __name__ == '__main__':
	u1 = Unet()
	u1.train(lr=1e-4,num_epochs=1)
	u1.continue_training(lr=1e-4,num_epochs=20)
	u1.continue_training(lr=1e-5,num_epochs=20)
	# u1.get_outputs()
	# u1.test_model()
	# u1.generate_masks()
