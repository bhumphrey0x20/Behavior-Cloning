import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Flatten, merge, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
import tensorflow as tf
from keras import backend as k

EPOCHS = 4

'''
	read csv file, load data into array. 
	This code taken from lecture
'''

data_path = './data'
data_path_1 = './data/jerky'
data_path_2 = './data/bridge'
data_path_3 = './data/jungle'

data= [data_path, data_path_1, data_path_2, data_path_3]

left_cam_Flag = 0
right_cam_Flag = 0

images=[]	
measurements=[]
for dat in data:
	lines=[]
	with open(dat + '/driving_log.csv') as file:
		reader = csv.reader(file)
		for line in reader:
			lines.append(line)
	
	for line in lines:
		source_path = line[0]
	#	source_path2 = line[1]
		filename = source_path.split('/')[-1]
		image_path = dat + '/IMG/'+ filename
		image = cv2.imread(image_path)
#		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#		image = cv2.Canny(image, 200,250)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)
		if left_cam_Flag > 0:
			source_path2 = line[1]
			filename = source_path2.split('/')[-1]
			image_path = dat + '/IMG/'+ filename
			image = cv2.imread(image_path)
			images.append(image)
			measurements.append(measurement)
		if right_cam_Flag > 0:
			source_path2 = line[2]
			filename = source_path2.split('/')[-1]
			image_path = dat + '/IMG/'+ filename
			image = cv2.imread(image_path)
			images.append(image)
			measurements.append(measurement)

X_train = np.array(images) #.reshape(-1,160,320,1)
y_train = np.array(measurements)

print('Done')	

Input = Input(shape=(160,320,3))
x = Lambda(lambda x: (x/255.) - 0.5)(Input)
x= Cropping2D(cropping= ((50,20), (0,0)), input_shape=(3,160,320))(x)

layer1 = Convolution2D(4, 3,3, border_mode='valid', activation='relu', name='c1')(x)
layer1 = MaxPooling2D(pool_size=(2,2), border_mode='valid', name='mx_pl1')(layer1)

layer2 = Convolution2D(32, 5,5, border_mode='valid', activation='relu',name='c2')(layer1)
layer2 = MaxPooling2D(pool_size=(2,2), border_mode='valid', name='mx_pl2')(layer2)

layer3 = Convolution2D(300, 5,5, border_mode='valid', name='c3')(layer2)

flat_a = Flatten( )(layer2)
flat_b = Flatten( )(layer3)

layer3 = merge([flat_a, flat_b], mode='concat', concat_axis=1)

logits = Dense(1, name='d4')(layer3)
print(logits.get_shape() )



# compile model and get loss function
model = Model(input=Input, output=logits)
model.compile(loss='mse', optimizer='adam')
history_loss = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch= EPOCHS)

model.save('model.h5')

print(history_loss.history.keys())
plt.plot(history_loss.history['loss'])
plt.plot(history_loss.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()	
