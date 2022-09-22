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


FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
model = load_model("C:\\Users\\errar\\Downloads\\arjun1_model.h5")


new_title = '<h1 style="font-family:monospace; margin-top:0;text-align:center; color:orange; font-size: 70px;">Sign Language Detection</h1>'
st.markdown(new_title, unsafe_allow_html=True)
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
    title = '<h2 style=" margin-top:0;text-align:center; color:red; font-size: 50px;">Camera cannot be initialised</h2>'
    st.markdown(title, unsafe_allow_html=True)


