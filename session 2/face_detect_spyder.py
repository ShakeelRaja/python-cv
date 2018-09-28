#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:46:23 2018

@author: sraja
"""


import matplotlib.pyplot as plt
import cv2 
import cvutils

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# imagePath = 'mona.jpg'
imagePath = 'beatles.png'
#imagePath = 'group1.jpg'
cascPath = "eyes.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cvutils.imshow("Original", image)
# cvutils.imshow("Greyscale", gray)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE # cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))


# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

cvutils.imshow("Faces found", image)
cv2.imwrite('out.jpg', image)