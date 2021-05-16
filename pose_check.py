import numpy as np
import os
import matplotlib.pyplot as plt
import pose_estimator
import json

with open('id_to_pose.json') as file:
    id_to_pose = json.load(file)
    
with open('gound_truth_angles.json') as json_file:
    ground_truth = json.load(json_file)
    
def error_check(ground_truth,pose_angles):  
    
    error = np.absolute(np.subtract(np.array(ground_truth),np.array(pose_angles)))
    if len(error[error<20])==len(error) :
        return "Perfect!" #Green
    elif  len(error[error<60 ])==len(error) :
        return "Put more effort!" #Yellow 
    else :
        return "Doing it wrong!!" #Red
    
def prediction(indx,image):
    
    test_angles = pose_estimator.get_angle_dict(image)
    ground_truth_pose = ground_truth[id_to_pose[indx]]
    error = error_check([*ground_truth_pose.values()],[*test_angles.values()])
    
    return error, id_to_pose[indx]