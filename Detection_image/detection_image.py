#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:57:59 2021

@author: prashant
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

labels = open('../Yolov3/coco.names').read().strip().split('\n')
# print(labels)

weights_path = "../Yolov3/yolov3.weights"
config_path = "../Yolov3/yolov3.cfg"
prob_min = 0.5
thresh = 0.3

net = cv.dnn.readNetFromDarknet(config_path, weights_path)
layers_names = net.getLayerNames()
# print(layers_names)
layers_names_output = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(layers_names_output)

img = cv.imread('input/img1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# print(img.shape)

plt.imshow(img)
plt.show()
# cv.imshow('img', img)
# cv.waitKey(0)

blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=False, crop=False)

print(blob.shape)

blob_ = blob[0, :, :, :].transpose(1, 2, 0)
print(blob_.shape)

# blob_ = cv.cvtColor(blob_, cv.COLOR_BGR2RGB)
# plt.imshow(blob_)
# plt.show()

net.setInput(blob)
output_net = net.forward(layers_names_output)
print(output_net[0].shape)

np.random.seed(42)
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

bounding_boxes = []
confidences = []
class_numbers = []

h, w = img.shape[:2]
print(h, w)

for result in output_net:
    # print(result.shape)
    for detection in result:
        # print(x.shape)
        scores = detection[5:]
        class_id = np.argmax(scores)
        current_confidence = scores[class_id]

        if current_confidence > prob_min:
            current_box = detection[0:4] * np.array([w, h, w, h])
            x_center, y_center, width, height = current_box.astype('int')
            x_min = (x_center - width/2)
            y_min = (y_center - height/2) 
            bounding_boxes.append([int(x_min), int(y_min), int(width), int(height)])
            confidences.append(float(current_confidence))
            class_numbers.append(class_id)

results = cv.dnn.NMSBoxes(bounding_boxes, confidences, prob_min, thresh)

for i in class_numbers:
    print(labels[i])

if len(results) > 0:
    for i in results.flatten():
        # print(i)
        x_min, y_min, box_width, box_height = bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][2], bounding_boxes[i][3]
        col = [int(j) for j in colours[class_numbers[int(i)]]]
        cv.rectangle(img, (x_min, y_min), (x_min+box_width, y_min+box_height), col, 2)
        text = '{}: {:.4f}'.format(labels[class_numbers[int(i)]], confidences[i])
        print(text)
        cv.putText(img, text, (x_min, y_min-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

# cv.imshow('Image', img)
# cv.waitKey(0)
cv.imwrite('output/outputImg1.jpg',cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.imshow(img)
plt.show()











