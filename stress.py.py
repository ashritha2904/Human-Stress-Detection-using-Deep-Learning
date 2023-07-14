#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[18]:


from scipy.spatial import distance as dist

from imutils.video import VideoStream
from imutils import face_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imutils
import time
import dlib
import cv2
tf.gfile=tf.io.gfile


# # To find the distance bw left and right eye

# In[19]:


def eye_brow_distance(leye,reye):
    global points
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    return distq


# # To find the distance between upper and lower lips

# In[20]:


def lpdist(l_lower,l_upper):
    lipdist = dist.euclidean(l_lower, l_upper)
    points_lip.append(int(lipdist))
    return lipdist


# # NORMALIZATION

# In[21]:


def normalize_values(points,disp,points_lip,dis_lip):
    normalize_value_lip = abs(dis_lip - np.min(points_lip))/abs(np.max(points_lip) - np.min(points_lip))
    normalized_value_eye = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    normalized_value =( normalized_value_eye + normalize_value_lip)/2
    stress_value = np.exp(-(normalized_value))
    print(stress_value)
    if stress_value>=0.60:
        return stress_value,"HIGH STRESS"
    else:
        return stress_value,"LOW STRESS"


# # To find the facial emotion

# In[22]:


def emotion_finder(faces,frame):
    global emotion_classifier
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ['scared','sad','angry','disgust']:
        label = 'STRESSED'
    else:
        label = 'NOT STRESSED'
    return label


# # To get the location of the eyes

# In[23]:


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the vertical landamrks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio(to chcek whether eye is opened)
    eye_opening_ratio = (A + B) / (2.0 * C)
   # blink.append(int(eye_opening_ratio))

    # return the eye aspect ratio
    return eye_opening_ratio


# In[24]:


# the consecuting frame factor tells us to consider this amount of frame.
ar_thresh = 0.3
eye_ar_consec_frame = 3
counter = 0
total = 0


# # To get the frontal face detector and shape predictor

# In[25]:


detector = dlib.get_frontal_face_detector()


# # To extract the key facial features

# In[26]:


predictor = dlib.shape_predictor("G:\data 1\Stress detection\Stress-Detection-master\shape_predictor_68_face_landmarks.dat")


# # Loading the trained model

# In[27]:
emotion_classifier = load_model("G:\data 1\Stress detection\Stress-Detection-master\_mini_XCEPTION.102-0.66.hdf5", compile=False)


# In[28]:


cap = cv2.VideoCapture(0)

points = []
points_lip = []
blink = []


# In[29]:


while(True):
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500,height=500)
    
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (l_lower, l_upper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
  
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    #preprocessing the image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detections = detector(gray,0)
    for detection in detections:
        emotion = emotion_finder(detection,gray)
        cv2.putText(frame, emotion, (20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        shape = predictor(frame,detection)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lBegin:lEnd]
        right_eye = shape[rBegin:rEnd]
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]
        openmouth = shape[l_lower:l_upper]

        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull= cv2.convexHull(right_eye)
        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)
        openmouthhull = cv2.convexHull(openmouth)

        cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [openmouthhull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        #calculating the EAR
        left_eye_Ear = eye_aspect_ratio(left_eye)
        right_eye_Ear = eye_aspect_ratio(right_eye)
        
        lipdist = lpdist(openmouthhull[-1],openmouthhull[0])
        distq = eye_brow_distance(leyebrow[-1],reyebrow[0])
        
        avg_Ear = (left_eye_Ear + right_eye_Ear)/2.0

        if avg_Ear<ar_thresh:
            counter+=1
        else:
            if counter>eye_ar_consec_frame:
                total+= 1
            counter = 0
        cv2.putText(frame, "BLINKS : {}".format(total), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        stress_value,stress_label = normalize_values(points,distq,points_lip,lipdist)
        cv2.putText(frame, "STRESS :{}".format(stress_label), (300, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame,"STRESS LEVEL:{}".format(str(int(stress_value*100))),(20,50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("STRESS DETECTION", frame)
    
# Exit when escape is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


# # Destroying windows

# In[13]:


cv2.destroyAllWindows()
cap.release()


# # Graph

# In[14]:


plt.plot(range(len(points)),points,'ro')
plt.title("Stress Level")
plt.show()


# In[ ]:





# In[ ]:




