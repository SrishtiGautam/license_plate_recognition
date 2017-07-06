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
	lpd = lp_localization(image)

	#Character segmentation and recognition
	characters = charac_segmentation(lpd)

if __name__ == "__main__":
    main(sys.argv[1])
