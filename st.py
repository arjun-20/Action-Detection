# import module
import streamlit as st
import cv2
import time
import numpy as np

import tensorflow
import tensorflow as tf

font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2

from keras.models import load_model
# Title
st.title("Hello ")





FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
model = load_model("DeepVisionModel.h5")

if(camera.isOpened()):
  st.title("Webcam Live Feed")
  run = st.checkbox('Run')
  while run:
      _, frame1 = camera.read()
      time.sleep(0.5)
      _, frame2 = camera.read()


      image_1_b_w = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
      image_2_b_w = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

      #image_1_b_w = np.dstack((image_1_b_w, image_1_b_w))
      #image_2_b_w = np.dstack((image_2_b_w, image_2_b_w))


      image_1_b_w = cv2.resize(image_1_b_w, (256,256))
      image_2_b_w = cv2.resize(image_2_b_w, (256,256))

      absdiff = cv2.absdiff(image_1_b_w, image_2_b_w)

      absdiff = np.dstack([absdiff, absdiff, absdiff])
      #st.title(absdiff.shape)

      FRAME_WINDOW.image(absdiff)

      absdiff1 = np.expand_dims(absdiff, axis = 0)


      #cv2_imshow(absdiff)
      val = model.predict(absdiff1)
      if val == 1:
          absdiff = cv2.putText(absdiff, 'Unsigned', org, font, 
                     fontScale, color, thickness, cv2.LINE_AA)
          FRAME_WINDOW.image(absdiff)

      else:
          absdiff = cv2.putText(absdiff, 'Signed', org, font, 
                     fontScale, color, thickness, cv2.LINE_AA)
          FRAME_WINDOW.image(absdiff)




  else:
      st.write('Stopped')

else:
  st.title("Camera cannot be initialised")


