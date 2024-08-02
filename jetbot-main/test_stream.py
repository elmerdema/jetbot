from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import os

app = Flask(__name__)
socketio = SocketIO(app)

# Change camera indices here.
# Assuming you have two cameras, change indices accordingly
cap1 = cv2.VideoCapture(0)
cap2 = cap1

def gen():
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Resize frames to same dimensions if needed
        frame1 = cv2.resize(frame1, (320, 240))
        frame2 = cv2.resize(frame2, (320, 240))

        # Display frames side by side here. Make changes in this line if orientation/res is to be changed
        side_by_side_frame = cv2.hconcat([frame1, frame2])

        ret, jpeg = cv2.imencode('.jpg', side_by_side_frame)
        if not ret:
            break

        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')


@app.route('/')
def index():
    """Render a basic HTML page to display video feed."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Route to stream video."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=3002, debug=True)
