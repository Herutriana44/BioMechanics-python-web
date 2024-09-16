from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from models.niosh_lifting_model import calculate_niosh_lifting_equation
from angle_calc import angle_calc

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        pose1=[]
        
        if results.pose_landmarks:
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z=[]
                h, w,c = image_rgb.shape
                x_y_z.append(lm.x)
                x_y_z.append(lm.y)
                x_y_z.append(lm.z)
                x_y_z.append(lm.visibility)
                pose1.append(x_y_z)
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id%2==0:
                    cv2.circle(image_rgb, (cx, cy), 5, (255,0,0), cv2.FILLED)
                else:
                    cv2.circle(image_rgb, (cx, cy), 5, (255,0,255), cv2.FILLED)
            niosh_score = calculate_niosh_lifting_equation(results.pose_landmarks, frame.shape)
            rwl = niosh_score['RWL']
            li = niosh_score['LI']
            cv2.putText(frame, f"RWL: {niosh_score['RWL']:.2f} kg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"LI: {niosh_score['LI']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        try:
            rula,reba=angle_calc(pose1)
            print(rula,reba)
            if (rula != "NULL") and (reba != "NULL"):
                if int(rula)>3:
                    message1 = ("Rapid Upper Limb Assessment Score : "+rula+" Posture not proper in upper body. ")
                    message2 = ("Posture not proper in upper body")
                    msg = message1# + message2
                    cv2.putText(frame, msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0, 255), thickness=2, circle_radius=4))
                else:
                    message1 = ("Rapid Upper Limb Assessment Score : "+rula)
                    cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if int(reba)>4:
                    message1 = ("Rapid Entire Body Score : "+reba+" Posture not proper in your body. ")
                    message2 = ("Posture not proper in your body")
                    msg = message1# + message2
                    cv2.putText(frame, msg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0, 255), thickness=2, circle_radius=4))
                else:
                    message1 = ("Rapid Entire Body Score : "+reba)
                    cv2.putText(frame, message1, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # root.update()
            else:
                message1 = ("Posture Incorrect")
                cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        except Exception as e:
            print(f"error : {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
