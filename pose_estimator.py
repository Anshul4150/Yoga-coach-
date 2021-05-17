import json 
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from collections import OrderedDict

import numpy as np
import os
import matplotlib.pyplot as plt

def estimate_keypoints(image):

    with mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.5) as pose:
        
        #image = cv2.imread(image_path)
        #print(image.shape)
        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            keypoints = results.pose_landmarks.landmark
        except:
            print("cant estimate keypoints")
            
        return keypoints
    
def draw_keypoints(image):
    
    with mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.5) as pose:
        
        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return annotated_image
    
def angle_between(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def get_angle_dict(image):
    
    d = OrderedDict()
    try:
        landmarks  = estimate_keypoints(image)
            
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        l_knee =  [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        
        d['l_wrist_ankle_shoulder'] = angle_between(l_wrist,l_ankle,l_shoulder)
        d['r_wrist_ankle_shoulder'] = angle_between(r_wrist,r_ankle,r_shoulder)
        d['l_elbow_shoulder_hip'] = angle_between(l_elbow ,l_shoulder ,l_hip)
        d['r_elbow_shoulder_hip'] = angle_between(r_elbow ,r_shoulder ,r_hip)
        d['l_hip_knee_ankle'] = angle_between(l_hip,l_knee, l_ankle)
        d['r_hip_knee_ankle'] = angle_between(r_hip,r_knee, r_ankle)
    except:
        print("can't estimate angle dictionary")
        return False 
        
    return d