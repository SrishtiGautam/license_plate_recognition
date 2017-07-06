from skimage import io,color,filters,measure
from skimage.morphology import disk,dilation
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import numpy as np

def lp_localization(image):

	print("Detecting license plate...")

	# Convert to grayscale
	gray_image = color.rgb2gray(image)

	# Mean filtering
	selem = disk(2)
	gray_image = filters.rank.mean(gray_image, selem=selem)

	# Vertical sobel edge detector
	sobel_image = abs(filters.sobel_v(gray_image))
	thresh = filters.threshold_otsu(sobel_image)
	binary = sobel_image > thresh

	# Dilation
	selem = disk(2)
	binary = dilation(binary,selem=selem)

	#remove small objects
	label_objects, nb_labels = ndi.label(binary)
	sizes = np.bincount(label_objects.ravel())
	mask_sizes = sizes > 1000
	mask_sizes[0] = 0
	cleaned_image = mask_sizes[label_objects]

	#remove large objects
	label_objects, nb_labels = ndi.label(cleaned_image)
	sizes = np.bincount(label_objects.ravel())
	mask_sizes = sizes < 5000
	mask_sizes[0] = 0
	cleaned_image = mask_sizes[label_objects]

	binary = cleaned_image

	# Label the different objects in the image
	label_img = measure.label(binary)

	# Find all objects
	regions = measure.regionprops(label_img)
	if(len(regions)>0):
		#License Plate detection based on aspect ratio of Bounding box
		for comp in range(0,len(regions)):
			sub_image = regions[comp].image
			if((sub_image.shape[1]/sub_image.shape[0])>2.5):
				r0, c0, r1, c1 = regions[comp].bbox
				final_detected_area = image[r0:r1, c0:c1]
				return final_detected_area

	else:
		return 0
