import os
folder = input("folder name:")  
os.mkdir(folder) #creating the folder 
import cv2
import numpy as np

video = input("enter video name:")
vidcap = cv2.VideoCapture(video)#capturing the video
count = 0
images=[]
while True:
    success,image = vidcap.read()
    if not success:
        break
    images.append(image)#creating an array of all frames
    #cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # saving all the frames
    count += 1
x = int(count/100)
need=np.array(images[::x]) #taking only 5th frame
count1 = 0
#while count1<len(need):
#    count1 += 1
for i in need:
    count1 += 1
    cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count1)),i) #writting all the n th frames



