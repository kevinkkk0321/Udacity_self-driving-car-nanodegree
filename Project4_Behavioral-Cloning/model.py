import os
import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten, Dense, Lambda , Cropping2D
from keras.layers.core import Dropout

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Model
import matplotlib.pyplot as plt
import pickle  

csv_data = []
data_path = "data/"
image_path = data_path + "IMG/"
correction = 0.2

with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # Skipping the headers
    next(reader, None)
    for line in reader:
        csv_data.append(line)
print('csv_data len:', len(csv_data))           
        
# Method to pre-process the input image
def pre_process_image(image):
    # Since cv2 reads the image in BGR format and the simulator will send the image in RGB format
    # Hence changing the image color space from BGR to RGB
    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Cropping the image
    cropped_image = colored_image[60:140, :]
    # Downscaling the cropped image
    resized_image = cv2.resize(cropped_image, None, fx=0.25, fy=0.4, interpolation=cv2.INTER_CUBIC)
    return resized_image
    
    
def generator(input_data, batch_size, train=False):
    num_data = len(input_data)
    processing_batch_size = int(batch_size/4)
    while 1: # Loop forever so the generator never terminates
        shuffle(input_data)
        for offset in range(0, num_data, batch_size):
            batch_data = input_data[offset:offset+processing_batch_size]
            image_data = []
            steering_angle = []
            for each_batch in batch_data:
                # Processing the center image
                center_image_path = image_path + each_batch[0].split('/')[-1]
                center_image = cv2.imread(center_image_path)
                steering_angle_for_centre_image = float(each_batch[3])
                if center_image is not None:
                    image_data.append(center_image)
                    steering_angle.append(steering_angle_for_centre_image)
                    if train == True:
                        # Flipping the image
                        image_data.append(cv2.flip(center_image, 1))
                        steering_angle.append(- steering_angle_for_centre_image)
                # Processing the left image
                left_image_path = image_path + each_batch[1].split('/')[-1]
                left_image = cv2.imread(left_image_path)
                if left_image is not None:
                    image_data.append(left_image)
                    steering_angle.append(steering_angle_for_centre_image + correction)
                    if train == True:
                        # Flipping the image
                        image_data.append(cv2.flip(left_image, 1))
                        steering_angle.append(- (steering_angle_for_centre_image + correction))
                # Processing the right image
                right_image_path = image_path + each_batch[2].split('/')[-1]
                right_image = cv2.imread(right_image_path)
                if right_image is not None:
                    image_data.append(right_image)
                    steering_angle.append(steering_angle_for_centre_image - correction)
                    if train == True:
                        # Flipping the image
                        image_data.append(cv2.flip(right_image, 1))
                        steering_angle.append(- (steering_angle_for_centre_image - correction))
            X_train = np.array(image_data)    
            y_train = np.array(steering_angle)    
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32
# split data
train_data, validation_data = train_test_split(csv_data, test_size=0.2)
# compile and train the model using the generator function
train_generator = generator(train_data, batch_size=batch_size, train=True)
validation_generator = generator(validation_data, batch_size=batch_size)

# Preprocess incoming data, centered around zero with small standard deviation 
model = Sequential()
model.add(Cropping2D(cropping=((50, 30), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # Normalization
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.50))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_data), \
                    validation_data=validation_generator, nb_val_samples=len(validation_data), \
                    nb_epoch=5, verbose=1)


model.save('model.h5')
### print the keys contained in the history object  
print(history_object.history.keys())

with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history_object.history, file_pi)
### plot the training and validation loss for each epoc
# plt.plot(history_object.history['loss)
# plt.plot(history_object.history['val_lo'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared err loss')
# plt.xlab('epoch')
# plt.legend(['training set', 'validation set'], loc='per right')
# plt.show()
