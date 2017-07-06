from lp_localization import lp_localization
from character_segmentation import charac_segmentation
import matplotlib.pyplot as plt
import sys
from skimage import io

def main(image_path):
	# lpd = lp_localization('../data/license plate images/41.JPG')
	# read image from directory
	image = io.imread(image_path)

	#License plate detection
	plt.subplot2grid((3,10),(0,0),colspan=7)
	plt.imshow(image)
	plt.axis('off')
	plt.title('Original Image')
	lpd = lp_localization(image)
	plt.subplot2grid((3,10),(1,0),colspan=7)
	plt.imshow(lpd)
	plt.axis('off')
	plt.title('Localized license plate')

	#Character segmentation and recognition
	characters = charac_segmentation(lpd)
	plt.show()

if __name__ == "__main__":
    main(sys.argv[1])
