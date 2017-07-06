import cv2
import numpy as np
import license_plate_recognition
from lp_localization import lp_localization
from character_segmentation import charac_segmentation

#capture image
cap = cv2.VideoCapture(0)

#Template license plate
template = cv2.imread('../Data/template.png',0)
w, h = template.shape[::-1]
while(True):
    ret, frame = cap.read()
    i= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(i,template,cv2.TM_CCOEFF_NORMED)
    cv2.imshow('frame',i)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #if high cross coreelation exists
    if(max_val>0.5):
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[0] + h)
        cv2.rectangle(i, top_left, bottom_right, (50, 0, 130), 2)
        s, im = cap.read() 
        
        lpd = lp_localization(im)
        if(lpd!=0):
            characters = charac_segmentation(lpd)
        else:
            print("No license plate found.")


cap.release()
cv2.destroyAllWindows()