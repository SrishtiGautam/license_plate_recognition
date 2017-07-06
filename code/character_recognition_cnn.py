import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Activation
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import h5py

K.set_image_dim_ordering('th')

batch_size = 80
input_size = (3,28,28)
nb_classes = 36

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()
                                  

train_generator = train_datagen.flow_from_directory(
        '../data/character_recog/train',  
        batch_size=batch_size,
        shuffle=True,
        target_size=input_size[1:],
        class_mode='categorical')  

validation_generator = test_datagen.flow_from_directory(
        '../data/character_recog/test',  
        batch_size=batch_size,
        target_size=input_size[1:],
        shuffle=True,
        class_mode='categorical')



# define the model
def cnn_model():
	# create model
	model = Sequential()
	model.add(Conv2D(20, (5, 5), input_shape=input_size, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(50, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dense(nb_classes, activation='softmax'))

	sgd = SGD(lr=0.001, decay=1e-2, momentum=0.9, nesterov=True)

	# Compile model
	model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
	print(model.summary())
	return model

# build the model
model = cnn_model()

# Fit the model
model.fit_generator(train_generator,
                                        steps_per_epoch=1656,
                                        validation_data=validation_generator,
                                        validation_steps=324,
                                        epochs=50,
                                        verbose=1)


#save model
model.save('../models/lenet_trained.h5')



