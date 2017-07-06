import numpy
from keras.models import load_model
import h5py

def predict_from_cnn(image):
	class_names = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	model = load_model('../models/lenet_trained.h5')
	predicted_class = model.predict_classes(image,verbose=0)
	return class_names[int(predicted_class)]







