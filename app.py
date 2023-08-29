from flask import Flask, render_template, Response
import cv2
import os
import json
import classifier
import pose_check
import matplotlib.pyplot as plt

app = Flask(__name__)

cap = cv2.VideoCapture("arun.mp4")  # use 0 for web camera
cap.set(3,640)
cap.set(4,480)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2


def gen_frames():  # generate frame by frame from camera

    
    counter = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        else:
            
            if counter % 8 == 0 or counter  == 0 :
                frame = cv2.resize (frame, (640,480), interpolation = cv2.INTER_NEAREST)
                pose_idx = classifier.get_pose(frame)  # Image to class
                feedback ,pose = pose_check.prediction(pose_idx,frame)  # image,class to feedback, pose_name
                if feedback == "None":
                    print("Person Not Detected, Please Try Again!!")
                frame = pose_check.pose_estimator.draw_keypoints(frame) # Draw skeleton
                frame = cv2.putText(frame, feedback, org, font, fontScale, color, thickness, cv2.LINE_AA)

                print(feedback)
                print(pose)
#                 cv2.imshow('demo', canvas)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
                
                if(pose != "None" and feedback != "None"):
                    pose_img= cv2.imread('data/'+ pose+ '.jpg') 
                    pose_img = cv2.resize (pose_img, (640,480), interpolation = cv2.INTER_NEAREST)
                    pose_img = cv2.putText(pose_img, pose, org, font, fontScale, color, thickness, cv2.LINE_AA)
                    
                    frame = cv2.hconcat([frame, pose_img])

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        counter += 1
            


#     cap.release()
#     cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)