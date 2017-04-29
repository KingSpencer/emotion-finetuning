from __future__ import division, print_function, absolute_import
import re
import numpy as np
from dataset_loader import DatasetLoader
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from constants import *
from os.path import isfile, join
import random
import sys


class EmotionRecognitionFine:

	def __init__(self, image, label):
    # self.dataset = DatasetLoader()
    		self._image = image
    		self._label= label

	def build_network(self):
    # Smaller 'AlexNet'
    # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
		print('[+] Building CNN')
		self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1])
		self.network = conv_2d(self.network, 64, 5, activation = 'relu')
		self.network = max_pool_2d(self.network, 3, strides = 2)
		self.network = local_response_normalization(self.network)
		self.network = conv_2d(self.network, 64, 5, activation = 'relu')
		self.network = max_pool_2d(self.network, 3, strides = 2)
		self.network = local_response_normalization(self.network)
		self.network = conv_2d(self.network, 128, 4, activation = 'relu')
		self.network = dropout(self.network, 0.3)
		self.network = fully_connected(self.network, 3072, activation = 'relu')
		self.network = fully_connected(self.network, 8, activation = 'softmax', restore = False)
		self.network = regression(self.network,
			optimizer = 'momentum',
      		loss = 'categorical_crossentropy',
      		learning_rate = 0.005)
		self.model = tflearn.DNN(
      		self.network,
      		checkpoint_path = SAVE_DIRECTORY + '/emotion_recognitionFine',
      		max_checkpoints = 1,
      		tensorboard_verbose = 2
    		)
		load_name = './data/Gudi_model_100_epochs_20000_faces'
		self.model.load(load_name)
		print('[!] Model loaded from %s' % load_name)

	def train(self):
		self.model.fit(self._image, self._label, n_epoch=40, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=30, snapshot_step=200,
          snapshot_epoch=False, run_id='model_finetuning')

	def save_model(self):
    		self.model.save('./data/fine_tuned_net')
    		print('[+] Model trained and saved as fine_tuned_net')

	def predict(self, image):
		if image is None:
			return None
		image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
		return self.model.predict(image)

if __name__ == "__main__":
	images = np.load('data_test.npy')
	labels = np.load('labels_test.npy')
	images = images.reshape([-1, 48, 48, 1])
	network = EmotionRecognitionFine(images, labels)
	network.build_network()
	network.train()
	network.save_model()

