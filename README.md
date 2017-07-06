# License plate recognition

This code is aimed at recognizing license plate number from images. 

### Prerequisites
Following modules need to be installed for running License plate recognition codes.
1.  Scikit-image
2.  Keras
3.  Theano
4.  h5py
5.  numpy
6.  matplotlib

Further, following module needs to be installed for real time License plate recognition via webcam
1.  OpenCV

## Directory structure
1.  Code: contains all python codes
2.  Data: contains all images used for training as well as testing.
3.  Model: contains the trained machine learning models

## Running the tests

### License plate recognition (LRP) system
Run python license_plate_recognition.py 'image path'
Example, python license_plate_recognition.py ../data/license\ plate\ images/55.JPG

### Real time License plate recognoition system
Run real_time_lpr.py 
(It will display license plate number as long as a valid license plate comes into the field of view)

## Data

1. Car images dataset has been downloaded from http://vision.ucsd.edu/belongie-grp/research/carRec/car_data.html
2. English Handwritten Characters dataset is downloaded from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz


## Algorithm Overview
### License plate localization
1. Convert into grayscale(for generalization).
2. Mean filtering(to remove noise).
3. Vertical sobel edge detector.
4. Morphological dilation.
5. Remove very small objects.
6. Remove very large objects.
7. Take bounding box arround each connected component.
8. Discard components based on aspect ratio of the bounding box.


### Character segmentation
1. Convert into grayscale(for generalization).
2. Otsu's binary thresholding.
3. Remove very small objects.
4. Remove very large objects.


### Character recognition
1. Take connected components in segmented image one by one starting from left.
2. Threshold segmented character via Otsu's thresholding.
3. Resize/Crop the character image to 28x28.
4. Predict the character from already trained CNN. (CNN architecture used: LeNet, Dataset used for training: English Handwritten Characters dataset http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz)
