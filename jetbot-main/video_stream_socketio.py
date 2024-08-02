from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
from jetcam.utils import bgr8_to_jpeg
from jetcam.csi_camera import CSICamera
import cv2
import os

os.system("echo $USER | sudo -S systemctl restart nvargus-daemon")

app = Flask(__name__)
socketio = SocketIO(app)

# Change camera indices here.
cap1 = CSICamera(capture_device=0, width=512, height=512)
cap2 = CSICamera(capture_device=1, width=512, height=512)

cap1.running = True
cap2.running = True

def gen():
    while True:
        frame1 = cap1.value
        frame2 = cap2.value

        # Display frames side by side here. Make changes in this line if orientation/res is to be changed
        side_by_side_frame = cv2.hconcat([frame1, frame2])

        ret, jpeg = cv2.imencode('.jpg', side_by_side_frame)
        if not ret:
            break

        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/engine', methods=['POST'])
def object_detection():
    if request.is_json:
        data = request.get_json()
        print(data)
    else:
        data = request.form.to_dict()
        print(data)

    return Response(status=200)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=3001, allow_unsafe_werkzeug=True)
