# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:27:32 2018

@author: madhu
"""

import cv2
import numpy as np

img = cv2.imread('night pic (2).png')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hsv = np.array(img_hsv, dtype='float32')
img_hsv1 = img_hsv[:,:,:]/255.
h, s, v = cv2.split(img_hsv1)
v_new = v*2.
img_hsv_new = cv2.merge([h, s, v_new]) 
img_bgr = cv2.cvtColor(img_hsv_new, cv2.COLOR_HSV2BGR)
cv2.imshow('bgr_new',img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
