from skimage import io,color,filters,measure,util
from skimage.morphology import disk,erosion
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import numpy as np
from character_recognition_cnn_predict import predict_from_cnn

def charac_segmentation(image):
	print("Extracting characters....")

	# Convert to grayscale
	gray_image = color.rgb2gray(image)

	# Otsu's thresholding
	thresh = filters.threshold_otsu(gray_image)
	binary = gray_image > thresh
	inverted_img = util.invert(binary)

	#remove small objects
	label_objects, nb_labels = ndi.label(inverted_img)
	sizes = np.bincount(label_objects.ravel())
	mask_sizes = sizes > 30
	mask_sizes[0] = 0
	cleaned_image = mask_sizes[label_objects]

	#remove large objects
	label_objects, nb_labels = ndi.label(cleaned_image)
	sizes = np.bincount(label_objects.ravel())
	mask_sizes = sizes < 500
	mask_sizes[0] = 0
	cleaned_image = mask_sizes[label_objects]

	# Label the different objects in the image
	label_img = measure.label(cleaned_image)

	# Find all objects
	regions = measure.regionprops(label_img)

	#get all centroids' column value
	centroid_col = np.zeros((len(regions),1))
	for comp in range(0,len(regions)):
		centroids = regions[comp].centroid
		centroid_col[comp] = centroids[1]

	sorted_indices = sorted(range(len(centroid_col)),key=centroid_col.__getitem__)

	license_no = ''

	print("Recognizing characters....")

	#Character recognition
	for comp in range(0,len(regions)):
		sub_image = regions[sorted_indices[comp]].image
		r0, c0, r1, c1 = regions[sorted_indices[comp]].bbox
		final_detected_area = image[r0:r1, c0:c1]
		final_charc = np.full((28,28),255,dtype=np.uint8)

		gray_image = color.rgb2gray(final_detected_area)

		#Threshold segmented character
		thresh = filters.threshold_otsu(gray_image)
		binary = gray_image > thresh
		binary_255 = np.empty((binary.shape),dtype=np.uint8)
		binary_255[binary==1] = 255
		binary_255[binary==0] = 0

		#Resize/Crop to 28x28
		if(final_detected_area.shape[0]<28):
			r0 = int(np.floor((28-final_detected_area.shape[0]+1)/2))
			r1 = int(final_detected_area.shape[0]+np.ceil((28-final_detected_area.shape[0])/2))
		else:
			r0 = 0
			r1 = 27
			binary_255 = binary_255[r0:r1,:]

		if(final_detected_area.shape[1]<28):
			c0 = int(np.floor((28-final_detected_area.shape[1]+1)/2))
			c1 = int(final_detected_area.shape[1]+np.ceil((28-final_detected_area.shape[1])/2))
		else:
			c0 = 0
			c1 = 27
			binary_255 = binary_255[:,c0:c1]

		
		final_charc[r0:r1,c0:c1] = binary_255
		rgb_image = color.gray2rgb(final_charc)
		plt.subplot2grid((3,10),(2,comp))
		plt.imshow(rgb_image)
		plt.axis('off')
		rgb_image = rgb_image.reshape([1,3,28,28])

		#Predict character from CNN
		predicted_char = predict_from_cnn(rgb_image)
		license_no = license_no + predicted_char
		plt.title(predicted_char)

	print(license_no)