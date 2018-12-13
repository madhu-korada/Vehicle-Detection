#detecting all the images in a folder
from yolo import YOLO

#from yolo import detect_image
from PIL import Image
import cv2
import os

import numpy as np                                                                    
import glob
img_folder = input('Input image foldername:')
#reading all the images
imagePath = glob.glob(img_folder + '/*.JPG') 
#creating an array of images of the folder
im_array = np.array( [np.array(Image.open(img).convert('L'), 'f') for img in imagePath] )
#creating new folder
folder = 'frames_11_detected'  
os.mkdir(folder) 

count = 0 #variable for loop

import scipy.misc


while count<len(im_array):
    count += 1
    for i in im_array:
        #detect the image
        rgb = scipy.misc.toimage(i)
        image,frame_info=YOLO.detect_image(YOLO(),rgb)
        
        #writting the image into new folder
        cv2.imwrite(os.path.join(folder,"frame_detected{:d}.jpg".format(count)),np.array(image))
   
    

