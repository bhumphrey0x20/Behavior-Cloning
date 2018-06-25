import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Input, Dense, Flatten, merge, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D, Dropout
import tensorflow as tf
from keras import backend as k

EPOCHS = 4
BATCH_SIZE = 125
alpha = int(np.floor(20000/BATCH_SIZE)) # 20000
SAMPLES_PER_EPOCH = alpha * BATCH_SIZE  
#SAMPLES_PER_EPOCH = BATCH_SIZE * 12
CHANNELS = 3
crop_amount = 20

rt_corr = 0 #-0.025		# amount to correct angles in recov
recover_angle = 0		# max angle in recovery data limit correction
angle_correction = -0.2


'''
	read csv file, load data into array. 
	This code was adapted from lecture
'''
USE_UD_Data = 1
USE_DATA_ADDON = 0
USE_RECOVER = 0

data_path_1 = './data'
data_path_2 = './data/jungle2'
data_path_3 = './data/turns'
data_addon = './data/rightside'
data_recover = './data/recover2'
addon_path =[data_addon] #, data_addon2]
data_path= [data_path_1, data_path_2]# 		, data_path_3,data_path_4]
new_measurements = []

# shifts image right or left by 'shift' number of pixels
# based on input argument translation - the shift direction: -1 shift left, +1 shift right
#
# ideas adapted from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

def image_shift(img, translation):
	shift_factor = 0.2
	shift = random.randint(5,20)
	shift = shift * shift_factor * translation
	dims = img.shape
	
	M = np.float32([ [1, 0, shift],[0, 1, 0] ] )
	img_shift = cv2.warpAffine(img, M, (dims[1],dims[0]))
	
	return img_shift, np.float(shift)


# randomly chooses to brighten or darken input image
# ideas adapted from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

def brighten_image(img):
	is_NonZero = 0
	
	while (is_NonZero == 0):
		bright_factor = np.random.uniform()
		if bright_factor != 0:
			is_NonZero = 1
	light_dark = random.randint(0,1)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)	
	hsv[:,:,2] = np.clip(hsv[:,:,2]*(light_dark +bright_factor), 0,255)
	img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return img	


# Reduces the amount of Zero steering angles by adding an randomized number to them

def preprocess_data(line_data, meas):
	#split data into left right center based on steering angle, similar to what is done here:
	#   https://github.com/ancabilloni/SDC-P3-BehavioralCloning/blob/master/model.py
	left_im, right_im, center_im = [],[],[]
	left_angle, right_angle, center_angle = [],[],[]

	data_mix, meas_mix = shuffle(line_data, meas)
	for i in np.arange(len(meas_mix)):
		angle = float(meas_mix[i])
		if (angle < -0.15 ):
			left_angle.append(angle)
			left_im.append(data_mix[i])
		elif( angle > 0.15 ):
			right_angle.append(angle)
			right_im.append(data_mix[i])
		else:
			center_angle.append(angle)
			center_im.append(data_mix[i])

	left_size = len(left_angle)
	right_size = len(right_angle)
	center_size = len(center_angle)
	# split center_im and center_angles, for redistro of center line driving
	center_im,  center_mix, center_angle, center_mix_angle = train_test_split(center_im, center_angle, test_size = 0.98)

	print('center_im', len(center_im))
	#print('center_mix', len(center_mix), '\n')
	
	for i in np.arange(len(center_mix)):
		angle = np.float(center_mix_angle[i])
		angle_coeff = np.random.uniform(0,0.25)
		if angle < 0:
			left_im.append(center_mix[i])
			left_im[-1].append(2)	# append flag to 'use right cam flag' 
			left_angle.append(angle - angle_coeff)
		if angle > 0:
			right_im.append(center_mix[i])
			right_im[-1].append(1) 	#append flag t0 'use left cam flag'
			right_angle.append(angle + angle_coeff)


	print('n_left  ', len(left_angle))
	print('n_center', len(center_angle))
	print('n_right ', len(right_angle))	
	new_lines = left_im  + right_im + center_im
	new_meas = left_angle + right_angle + center_angle 
#	new_lines, new_meas = reduce_centerline(line_data,meas)
	return new_lines, new_meas


#### Generator Functions

# generates image and associated steering angle. Converts image to YUV, crops top and bottom of
# image and resized to 64x64

def generate_data(data_lines, steering_angles ):
	crop_t = 50	#06/22/2018
	crop_b = 20	#06/22/2018
	line_size = len(data_lines)
	#print('line_size: %i' %(line_size))
	while True:
		#for i in range(len(data_lines)):		
			i = random.randint(0,line_size-1)
			if( len(data_lines[i]) == 8):
				cam = data_lines[i][-1] 	#used appended cam flag (right or left)
			else:
				cam = 0						# use center cam
			#cam = random.randint(0,2)
			source_path = data_lines[i][cam].lstrip().rstrip()
			image = cv2.imread(source_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
			'''
			if( USE_UD_Data == 1):
				filename = source_path.split('/')[-1]
				image_path = './IMG/'+ filename
				image = cv2.imread(image_path)			
				image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
			else:
				image = cv2.imread(source_path)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
			'''
			shape = image.shape
			image = image[crop_t:shape[0]-crop_b, 0:shape[1], :] 	#06/22/208
			image = cv2.resize(image, (64,64), interpolation= cv2.INTER_LINEAR)	#06/22/208
			yield image, steering_angles[i]
			
# Generates a batch of data base on batch_sz and randomly augments generated data images

def generate_batch(gen_data, batch_sz, gen_addon= None):

	steer = 0
	#batch_sz = int(batch_sz/2)
	while True:
		x_batch = []
		y_batch = []
		counter = 0
		if(gen_addon != None):
			b_size = (batch_sz)-1
		else:
			b_size = batch_sz 

		#print(b_size)

		for i in list(range(0,batch_sz)):
			select_augment= random.randint(0,4)
			if( counter < b_size):
				im = next(gen_data)
			else:
				im = next(gen_addon)

			# Flip Image
			if ( select_augment == 0 ):
				imflip = np.fliplr(im[0])
				x_batch.append(imflip)
				steer = float(im[1]) * -1
				y_batch.append(steer )
				new_measurements.append(steer)
			# Brighten or Darken Image
			elif( select_augment == 1 ):
				im_aug = brighten_image(im[0])
				x_batch.append(im_aug)
				steer = float(im[1])
				y_batch.append(steer)
				new_measurements.append(steer)				
			# Brighten/Darken Image and flip
			elif( select_augment == 2 ):
				im_eq = brighten_image(im[0])
				im_eq = np.fliplr(im_eq)
				x_batch.append(im_eq)
				steer = -1*float(im[1])
				y_batch.append(steer)
				new_measurements.append(steer)	
			# Shift Image
			elif( select_augment == 3 ):
				steer = float(im[1])
				if( steer >0):	translation = 1
				if(steer < 0):	translation = -1
				if(steer == 0): translation = 0
				im_aug, steer_corr = image_shift(im[0], translation)
				steer = steer + (translation * np.random.uniform() ) #(steer_corr/10.0)
				x_batch.append(im_aug)
				y_batch.append(steer)
				new_measurements.append(steer)	
			# No additional augmentation				
			else:
				x_batch.append(im[0])
				y_batch.append(float(im[1]))
				new_measurements.append(steer)		
			counter+=1
#		x_batch = np.array(x_batch).reshape(-1,160,320,CHANNELS)	#06/22/208
		x_batch = np.array(x_batch).reshape(-1,64,64,CHANNELS)
		#print(counter)
		yield x_batch, y_batch
		

images=[]	
measurements=[]
left_cam = []
left_cam_meas = []
right_cam = []
right_cam_meas = []
num_lines = 0
cnt = 0
lines=[]
		
if(USE_UD_Data == 0):
		
	for data in data_path:
		with open(data + '/driving_log.csv') as file:
			reader = csv.reader(file)
			for line in reader:
				# only read in every other line cut frame rate from 60 fps to 30
	#			if cnt%3 == 0:
				lines.append(line)
				measurements.append(line[3])
				cnt +=1
				#if (num_lines == MASK):	break;
	print('N_lines ', len(cnt))
else:	#use udacity simulator data
	
	#data = './UD/data'

	with open('./driving_log.csv') as file:
		reader = csv.reader(file)
		for line in reader:
			lines.append(line)
			measurements.append(line[3])
			cnt +=1
	print('N_lines ', cnt)
#measurements = np.float32(np.asarray(measurements))

	
lines_addon, measurements_addon = [],[]
if( USE_DATA_ADDON == 1 ):
	ln_cnt = 0	
	for data in addon_path:
		with open(data_addon + '/driving_log.csv') as file:
			reader = csv.reader(file)
			for line in reader:
				ln_cnt+=1
				lines_addon.append(line)
				angle = np.float32(line[3])
				
				if (angle < 0):
					angle += angle_correction
				'''
				if (ln_cnt > 1830) & (ln_cnt < 1920):
					if( angle >0):
						angle += rt_corr
				'''
				measurements_addon.append(angle)
				cnt +=1
	print('N_lines_addon ', ln_cnt)

recover_cnt = 0
lines_recover, measurements_recover = [],[]
if( USE_RECOVER ==1 ):
	
	with open(data_recover + '/driving_log.csv') as file:
		reader = csv.reader(file)
		for line in reader:
			angle = np.float32(line[3])
			
			if (angle < recover_angle):
				measurements_recover.append(angle + rt_corr)
				lines_recover.append(line)
				
			#measurements_recover.append(angle)
			#lines_recover.append(line)
			recover_cnt +=1
	print('N_lines_recover ', recover_cnt)
	#measurements_addon = np.float32(np.asarray(measurements_addon))
	#X_addon, y_addon = preprocess_data(lines_addon, measurements_addon)

if( USE_DATA_ADDON ==1):
	lines = lines + lines_addon
	
	measurements = measurements + measurements_addon
	#measurements = np.float32(np.asarray(measurements))
	
if( USE_RECOVER == 1):
	lines = lines + lines_recover	
	measurements = measurements + measurements_recover

measurements = np.float32(np.asarray(measurements))


print('lines_addon: %i' %(len(lines_addon)) )

print('cnt: %i' %(cnt) )
print('num lines: %i' %(len(lines)) )
print('num measurements %i' %(len(measurements)) )

"""
#plot histogram of steering angles from collected data before processing
bins = np.unique(measurements)
mx = np.amax(bins)
mn = np.amin(bins)
plt.hist(measurements, len(bins), [mn,mx])
plt.show()
"""
lines, measurements = preprocess_data(lines, measurements)

print('Num Lines', len(lines), 'Num Measurements', len(measurements))

#shuffle and split data for training and testing
X_train, X_valid, y_train, y_valid = train_test_split(lines, measurements, shuffle=True, test_size=0.2 )

train_size = len(y_train)
valid_size = len(y_valid)
print('train_size %i' %(train_size))
#alpha = int(np.floor(train_size/BATCH_SIZE))
#SAMPLES_PER_EPOCH= BATCH_SIZE*alpha


'''
if (USE_DATA_ADDON ==1):
	addon_data = generate_data(X_addon, y_addon)
else:
	addon_data = None
'''
train_data= generate_data(X_train, y_train)	
train_batch = generate_batch(train_data, BATCH_SIZE)
valid_data = generate_data(X_valid, y_valid)
valid_batch = generate_batch(valid_data, BATCH_SIZE)


"""
img=[]
#for i in range(BATCH_SIZE):
img = next(train_batch)
#img.append(im[0])

print(len(img))

plt.figure()
plt.subplot(2,2,1)
plt.imshow(img[0][0])
plt.subplot(2,2,2)
plt.imshow(img[0][1])
plt.subplot(2,2,3)
plt.imshow(img[0][8])
plt.subplot(2,2,4)
plt.imshow(img[0][9])
plt.show()
"""

print('Done')	


# my version of Nvidia CNN
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/

#Input = Input(shape=(160,320,CHANNELS))
Input = Input(shape=(64,64,CHANNELS))	#06/22/208
x = Lambda(lambda x: (x/255.) - 0.5)(Input)

# cropping to by 20 and bottom by 50 to remove extraineous background objects
# cropping left and right by 20 to account for preprocessing shift
#x= Cropping2D(cropping= ((50,20), (crop_amount,crop_amount)), input_shape=(CHANNELS,64,64))(x)	#06/22/208

layer1 = Convolution2D(24, 5,5, border_mode='valid', activation='relu', name='c1')(x)
layer1 = MaxPooling2D(pool_size=(2,2), border_mode='valid', name='mx_pl1')(layer1)


layer2 = Convolution2D(36, 5,5, border_mode='valid', activation='relu',name='c2')(layer1)
layer2 = MaxPooling2D(pool_size=(2,2), border_mode='valid', name='mx_pl2')(layer2)
#layer2 = Dropout(0.8)(layer2)


layer3 = Convolution2D(48, 5,5, border_mode='valid', name='c3')(layer2)
layer1 = MaxPooling2D(pool_size=(2,2), border_mode='valid', name='mx_pl1')(layer1)

layer4 = Convolution2D(64, 3,3, border_mode='valid', name='c4')(layer3)

layer5 = Convolution2D(64, 3,3, border_mode='valid', name='c5')(layer4)

# Dropouts and keep prob values from https://github.com/bhumphrey0x20/Behavior-Cloning/edit/master/writeup.md

f1 = Flatten( )(layer5)
f1 = Dropout(0.5)(f1)
f2 = Dense(1164)(f1)
f2 = Dropout(0.5)(f2)
f3 = Dense(100)(f2)
f3 = Dropout(0.5)(f3)
f4 = Dense(50)(f3)
f4 = Dropout(0.5)(f4)
f5 = Dense(10)(f4)

#layer3 = merge([flat_a, flat_b], mode='concat', concat_axis=1)
#layer3 = Dropout(0.8)(layer3)
logits = Dense(1)(f5)
#print(logits.get_shape() )



# compile model and get loss function
model = Model(input=Input, output=logits)
model.compile(loss='mse', optimizer='adam')

# uncomment line to load pretrained weights
#model.load_weights('model_d.h5_weights')

#history_loss = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch= EPOCHS)
history_loss = model.fit_generator(train_batch, samples_per_epoch=SAMPLES_PER_EPOCH, nb_epoch=EPOCHS, verbose=1, validation_data=valid_batch, nb_val_samples=SAMPLES_PER_EPOCH)



model.save('model_d.h5')

print(history_loss.history.keys())
plt.plot(history_loss.history['loss'])
plt.plot(history_loss.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()	



#plot histogram of steering angles from collected data
plt.figure()
bins = np.unique(measurements)
mx = np.amax(bins)
mn = np.amin(bins)
plt.hist(measurements, len(bins), [mn,mx])
plt.show()
"""

"""
# plot histogram of angles processed after batch generation for model
n_m_size = len(measurements)
nz = np.count_nonzero(measurements)
print('New Array len: %i' %(n_m_size))
print('Number of Zeros: %i' %(n_m_size -nz) )

"""
plt.figure()
n_bins = np.unique(measurements)
mx = np.amax(n_bins)
mn = np.amin(n_bins)
plt.hist(measurements, len(n_bins), [mn,mx])
plt.show()
"""

n_m_size = len(new_measurements)
nz = np.count_nonzero(new_measurements)
print('New Array len: %i' %(n_m_size))
print('Number of Zeros: %i' %(n_m_size -nz) )
