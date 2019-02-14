#!/usr/bin/env python
# coding: utf-8

# # Live Sentiment Analysis

# In[1]:


import cv2
import numpy as np
from keras.models import load_model


# In[2]:


#handler for camera
camera_handle = cv2.VideoCapture(0)

#loading pre trained haar code for face detection
face_haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

#loading saved cnn model
model = load_model('live_emotion_recognition_model.h5')

#predictiong face emotion using saved model
def get_emotion(image, shape_x, shape_y):
    #reshape to the size of face images of model trained on
    image = cv2.resize(image,(shape_x, shape_y))
    image = image.reshape((1, shape_x, shape_y, 1))
    result = model.predict_classes(image, verbose=0)
    emotions = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness', 4: 'Anger', 5: 'Disgust', 6: 'Fear',
               7: 'Contempt', 8: 'unknown', 9: 'Not a Face'}
    return emotions[result[0]]


# In[3]:


#main loop
while True:
    _, rgb_image = camera_handle.read()
    flipped_rgb_image = cv2.flip(rgb_image, 1)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    faces = face_haar.detectMultiScale(gray_image, 1.3, 5)#to tune
    
    for (x, y, w, h) in faces:
        gray_face = gray_image[y:y+h, x:x+w]
        #specify model face size here 48x48
        emotion = get_emotion(gray_face, 48, 48)
        cv2.rectangle(flipped_rgb_image,(x,y),(x+w,y+h),(0, 255, 0),2)
        cv2.putText(flipped_rgb_image, emotion, (30,30), font, 0.8, (0, 255, 0), 1)
    
    cv2.imshow('Live Emotion Recognition', flipped_rgb_image)
    
    k = cv2.waitKey(1) & 0xEFFFFF
    if k==27:   
        break
    else:
        continue
        
cv2.destroyAllWindows()

