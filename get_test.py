import os
import shutil
import numpy as np
import cv2

def emotion_to_vec(x):
    d = np.zeros(8)
    d[x] = 1.0
    return d

def get_image_label():
	images = []
	labels = []
	# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
	emotion_dict = {'angry':0, 'contempt':7, 'disgust':1, 'fear':2, 'happy':3, 'neutral':6, 'sadness':4, 'surprise':5}
	root = '/home/wangzf/emotion-recognition-neural-networks-master/dataset128'


	emotions = os.listdir(root)
	print emotions
	for emotion in emotions:
		im_dir = os.path.join(root, emotion)
		cur_images = os.listdir(im_dir)
		for img_name in cur_images:
			image = cv2.imread(os.path.join(im_dir, img_name))
			if len(image.shape) > 2 and image.shape[2] == 3:
				image = image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
			images.append(image)
			labels.append(emotion_to_vec(emotion_dict[emotion]))
	return np.array(images), np.array(labels)
images, labels = get_image_label()
print images.shape
print labels.shape

images = np.vstack((images, images[:, :, ::-1]))
labels = np.vstack((labels, labels))

print images.shape
print labels.shape
np.save('data_test.npy', images)
np.save('labels_test.npy', labels)
