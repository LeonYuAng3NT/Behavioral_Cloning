import csv
import cv2
from scipy import ndimage
import os, os.path
import numpy as np
from six.moves import cPickle as pickle

steering_angles = []
with open('./data/driving_log.csv', 'r') as csvfile:
	myreader = csv.reader(csvfile)
	for row in myreader:
		steering_angles.append(row[3])


center_angles = [float(i) for i in steering_angles]
left_angles = [(float(i)+.1*abs(float(i))+.05) for i in steering_angles]
right_angles = [(float(i)-.1*abs(float(i))-.05) for i in steering_angles]

left_angles = np.maximum(left_angles,-1.0)
left_angles = np.minimum(left_angles, 1.0)

right_angles = np.maximum(right_angles,-1.0)
right_angles = np.minimum(right_angles, 1.0)

steering_angles = np.concatenate([center_angles,left_angles,right_angles])

examples = len(steering_angles)

def load_images(num_images):

	name = os.listdir('./data/IMG')

	dataset = np.ndarray(shape=(num_images, 40, 80, 3),
                         	    dtype=np.float32)

	for image in range(num_images):
		image_file = './data/IMG/'+name[image]
		process_image = ndimage.imread(image_file)
		dataset[image,:,:,:] = cv2.resize(process_image,(80,40))

	print('Full dataset tensor:' , dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Max:', np.amax(dataset))
	print('Min:', np.amin(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset



def create_pickle(directory, force=True):
	filename = directory+'.p'
	if os.path.exists(filename) and not force:
		print('%s already present - Skipping pickling.' % filename)
	else:
		print('Pickling %s.' % filename)
		dataset = load_images(examples)
		try:
			with open(filename, 'wb') as f:
				pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', filename, ':', e)
	return filename

def merge_datasets(target_file, angles):
	angles_set = np.ndarray(examples, dtype=np.float32)
	try:
		with open(target_file, 'rb') as f:
			image_set = pickle.load(f)
			angles_set = angles
	except Exception as e:
		print('Unable to process data from', target_file, ':', e)
		raise

	try:
		f = open(target_file, 'wb')
		save = {
    			'image_set': image_set,
    			'angles_set': angles_set,
    		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print('Unable to save data to', target_file, ':', e)
		raise

pickle_name = 'train'
my_dataset = create_pickle(pickle_name)
merge_datasets(my_dataset, steering_angles)
