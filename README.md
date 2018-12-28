# Vehicle-Detection
In this project, the 2D vision data from a reference camera is taken as input. 

The camera was fixed on a truck and video data was acquired for a certain time under different weather and road conditions in India. 

The acquired video data had a resolution of 640x480 with 30 fps.

The objectives of the project with respect to Vehicle Detection are as follows:

- To detect (classify & localize) vehicles, pedestrians, traffic signals, animals, etc. in the given    sample videos. 

- To reduce the number of false negatives and false positives. 

- To achieve 90% mAP (mean Average Precision). 

- To increase robustness to get desired results in all scenes such as sunny, shadowed as well as in the dark. 

- To export the acquired frame data and write it in a .CSV file. 

# Deep learning frameworks

A Deep Learning Framework is a building block for designing, training and validating Deep Neural networks. Given that deep learning is the key to executing tasks of higher level of sophistication, building and deploying them successfully proves to be quite the herculean challenge for data scientists and data engineers across the globe. Today, we have a myriad of frameworks at our disposal that allows us to develop tools that can offer a better level of abstraction along with simplification of difficult programming challenges.

There are a variety of deep learning frameworks available. Each framework is built in a different manner for different purposes. They vary based on the languages they support, CNN modeling capability, RNN modeling capability, speed, etc. The comparison of the frameworks is given below.

![alt text](https://github.com/madhu-korada/img)

TensorFlow was chosen as the framework for vehicle detection based on the following factors:

- Python Interface

-	Well documented tutorials and references

- Increasing contributions on GitHub

- Excellent CNN Modelling capability

- Straight-forward & modular architecture

- Great speed with different datasets

- Multiple GPU support

- Keras compatible

# CNN Architecture

CNNs are Multilayer Neural Networks designed to recognize visual patterns directly from pixel images with minimal preprocessing. CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns the filters that in traditional algorithms were hand-engineered. This independence from prior knowledge and human effort in feature design is a major advantage. They have applications in image and video recognition, recommender systems and natural language processing.

There are different CNN architectures such as R-CNN, Fast R-CNN, Faster R-CNN, SPP-Net, YOLO, SSD which are used for object detection. There are two ways of approaching it:







-	Detection as a Classification Problem: R-CNN and its variants & SPP-Net.

- Detection as a Regression Problem: YOLO (You Only Look Once) and SSD (Single Shot Detector).

![alt text](https://github.com/madhu-korada/img)
![alt text](https://github.com/madhu-korada/img)


YOLO was chosen as the object detector in this case because of the following reasons:

-	State-of-the-art object detector at present

- Real-time object detection (30 FPS)

-	Good mean Average Precision (mAP) value

-	Good accuracy and speed over a variety of datasets 

-	Availability of pre-trained weights

![alt text](https://github.com/madhu-korada/img)

The YOLO v3 network Architecture was implemented using Darknet in Python. It is a 106 layer fully convolutional underlying architecture. Detection takes place at three scales - big objects, medium objects and small objects. It detects big objects as 13x13 grid at 82nd layer, medium objects as 26x26 grid at 94th layer and small objects as 52x52 grid at 106th layer.  There are 10,647 bounding boxes for a 416x416 image. It also supports multilabel classification.


# Dataset
It is a large collection of images which are annotated. These images are used for training the network, if it is not already pre-trained.  Each dataset will have a certain number of classes. Based on these classes, the datasets are chosen for appropriate applications. 

There are lots of datasets from which COCO dataset was chosen. It has 80 classes with more than 200,000 images. 18 classes considered relevant to driving in India were chosen from the overall 80 and the network was trained to detect only these classes.

![alt text](https://github.com/madhu-korada/img)

# Implementation

The algorithm was implemented in Python as it is simple as well as the best language to use with TensorFlow Framework. Spyder IDE from Anaconda (Python Distribution) was used for running the code.

