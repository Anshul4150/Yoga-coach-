from flask import Flask, render_template, Response
import cv2
import os
import json
import classifier
import pose_check
import matplotlib.pyplot as plt

app = Flask(__name__)

cap = cv2.VideoCapture(0)  # use 0 for web camera
cap.set(3,640)
cap.set(4,480)

def gen_frames():  # generate frame by frame from camera

    counter = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        else:
            counter += 1
            
            if counter % 3  == 0:

                pose_idx = classifier.get_pose(frame)
                feedback ,pred = pose_check.prediction(pose_idx,frame)
                if feedback == "None":
                    print("Person Not Detected, Please Try Again!!")
                canvas = pose_check.pose_estimator.draw_keypoints(frame)

                print(feedback)
                print(pose)
#                 cv2.imshow('demo', canvas)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break

                ret, buffer = cv2.imencode('.jpg', canvas)
                canvas = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + canvas + b'\r\n')
            


#     cap.release()
#     cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)