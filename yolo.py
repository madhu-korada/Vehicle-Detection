#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""
import cv2
import colorsys
import os
#from timeit import default_timer as timer

import glob
import scipy.misc

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

frame_no = 0

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5' # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        
        global frame_no
        frame_no+=1
        #obj_info = np.array([])
        print(frame_no)
        #start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        
        #preprocessing for night and day 
        image_mean = np.mean(image_data)
        if image_mean < 60 :
            img_hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
            img_hsv = np.array(img_hsv, dtype='float32')
            #img_hsv1 = img_hsv[:,:,:]/255.
            h, s, v = cv2.split(img_hsv)
            v_new = v*1.5
            img_hsv_new = cv2.merge([h, s, v_new]) 
            image_data = cv2.cvtColor(img_hsv_new, cv2.COLOR_HSV2RGB)
        
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        out_classes,out_scores,out_boxes=np.array(out_classes),np.array(out_scores),np.array(out_boxes)
        for q, w in reversed(list(enumerate(out_classes))):
            if w > 20 or w == 4 or w == 6 or w == 8 or w == 9 or w == 10 or w == 11 or w == 12 or w == 13  : 
                #i, = np.where(out_classes == j)
                #m_out_boxes = np.delete(out_boxes,q)
                m_out_scores = np.delete(out_scores,q)
                m_out_classes = np.delete(out_classes,q)
                out_scores, out_classes = m_out_scores, m_out_classes

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        #print(out_classes)
        #print(out_boxes)
        #print(out_scores)
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        frame_info = np.array([])
        no_of_bb = np.array(len(out_boxes))
        if no_of_bb == 0:
            frame_info = np.array([ frame_no, 0, 'None', 0, 0, 0, 0, 0 ])
        else:    
            for i, c in (list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                
                temp = np.array([frame_no, i+1, predicted_class, score, top, left, bottom, right])
                #print('temp=')
                #print(temp)
                if i==0 :
                    frame_info = temp
                else:
                    frame_info = np.vstack((frame_info,temp))
                
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # Image Drawing
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
        if frame_info.size == 0:
            frame_info = np.array([ frame_no, 0, 'None', 0, 0, 0, 0, 0 ])
        print('frame_info = ')
        print(frame_info)
        #end = timer()
        #print(end - start)
        return image, frame_info

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    #video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        #rint("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        #print(output_path,video_FourCC,video_fps,video_size)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, video_size)
    #accum_time = 0
    #curr_fps = 0
    #fps = "FPS: ??"
    #prev_time = timer()
    video_info = np.array([])
    while (vid.isOpened()):
        return_value, frame = vid.read()
        #print(return_value)
        if return_value == True :
            frame = np.asarray(frame)
            image = Image.fromarray(frame)
            image, f_info = yolo.detect_image(image)
            if frame_no == 1:
                video_info = f_info
            else:
                video_info = np.vstack((video_info, f_info))
            #print('video_info = ')
            #print(video_info)
            result = np.asarray(image)
            
            #curr_time = timer()
            #exec_time = curr_time - prev_time
            #prev_time = curr_time
            #accum_time = accum_time + exec_time
            #curr_fps = curr_fps + 1
            #if accum_time > 1:
            #    accum_time = accum_time - 1
            #    fps = "FPS: " + str(curr_fps)
            #    curr_fps = 0
            #    cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                fontScale=0.50, color=(255, 0, 0), thickness=2)
            
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
                              
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if isOutput:
                out.write(result)
        
        else :
            out.release()
            cv2.destroyAllWindows()
            break

    yolo.close_session()
    return video_info

def detect_img(yolo,img):
    while True:
        #img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, f_info = yolo.detect_image(image)
            #r_image = np.array(r_image)
            #cv2.imshow('Test image',r_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #r_image.show()
    yolo.close_session()
    return r_image
    
"""def detect_folder(yolo):
    while True:
        img_folder = input("input folder name:")
        imagePath = glob.glob(img_folder + '/*.JPG') 
        #creating an array of images of the folder
        im_array = np.array( [np.array(Image.open(img).convert('L'), 'f') for img in imagePath] )
        folder = input("output folder name:")  
        os.mkdir(folder) 
        count = 0 #variable for loop



        while count<len(im_array):
            count += 1
            for i in im_array:
            #detect the image
                rgb = Image.fromarray(i)
                r_image,f_info = yolo.detect_image(rgb)
            #image,frame_info=YOLO.detect_image(YOLO(),rgb)
        
            #writting the image into new folder
            cv2.imwrite(os.path.join(folder,"frame_detected{:d}.jpg".format(count)),r_image)"""
   


if __name__ == '__main__':
    #detect_img(YOLO())
    video_path = "5.4.mp4"
    output_path = "output_5.4.mp4" 
    v_info = detect_video(YOLO(), video_path, output_path)
    #print(v_info)
    np.savetxt('Video Info 5.4.csv', v_info, fmt='%s', delimiter=',', header="Frame No., Object No., Class, Conf. Score, Top, Left, Bottom, Right")