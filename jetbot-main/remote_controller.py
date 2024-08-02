import ipywidgets.widgets as widgets
from jetracer.nvidia_racecar import NvidiaRacecar
import traitlets
from jetcam.csi_camera import CSICamera
import os
import time
import threading
import logging
import socket
from IPython.display import display, Image
import ipywidgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg

# Create logging file
log = logging.getLogger("remote_controller")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('remote_controller.log')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

from data_log import save_data, remove_data

class MovementController:
    def __init__(self, throttle_gain=0.13, steering_offset=-0.12):
        # Initialize movement
        self.controller = widgets.Controller(index=0)
        self.car = NvidiaRacecar()
        self.car.steering = 0
        self.car.throttle_gain = throttle_gain
        self.car.steering_offset = steering_offset
        
    def show_controls(self):
        display(self.controller)
        
    def link_axis(self):
        self.left_link = traitlets.dlink((self.controller.axes[0], 'value'), (self.car, 'steering'))
        self.right_link = traitlets.dlink((self.controller.axes[3], 'value'), (self.car, 'throttle'), transform=lambda x: -x)
        
    def unlink_axis(self):
        self.left_link.unlink()
        self.right_link.unlink()
        
    # Getters
    def get_car(self):
        return self.car
      
    def get_controller(self):
        return self.controller
      
    def get_steering_angle(self):
        if abs(self.car.steering) < 0.01:
            return 0
        return self.car.steering
      
    def get_throttle(self):
        return self.car.throttle


class Recorder:
    def __init__(self, movement_controller):
        # Initialize cameras
        os.system("echo $USER | sudo -S systemctl restart nvargus-daemon")
        self.camera0 = CSICamera(capture_device=0, width=512, height=512)
        self.camera1 = CSICamera(capture_device=1, width=512, height=512)
        self.camera0.running = True
        self.camera1.running = True
        
        # Recording state
        self.recording = False
        self.movement_controller = movement_controller
        
        # Recording ID
        self.rec_id = 0
        
        # Create and start the thread for recording
        self.recording_thread = threading.Thread(target=self.start_recording)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Create and start the thread for stopping recording
        self.stop_thread = threading.Thread(target=self.stop_recording)
        self.stop_thread.daemon = True
        self.stop_thread.start()
        
    def start_recording(self):
        while True:
            A_button = self.movement_controller.controller.buttons[0].value
            if A_button == 1 and not self.recording:
                self.recording = True
                self.rec_id = max([int(d) for d in os.listdir('jetson_data') if d.isdigit()], default=0) + 1
                print(f"Recording started with ID {self.rec_id}")
                
                # Main loop for recording
                while self.recording:
                    left_cam_image = self.camera0.value
                    right_cam_image = self.camera1.value
                    steering_angle = self.movement_controller.get_steering_angle()

                    # Save the data
                    save_data(left_cam_image, right_cam_image, steering_angle, self.rec_id)

                    # Introduce a delay to avoid busy-waiting
                    time.sleep(0.1)
                
            time.sleep(0.1)
    
    def stop_recording(self):
        while True:
            B_button = self.movement_controller.controller.buttons[1].value
            if B_button == 1 and self.recording:
                self.recording = False
                print(f"Recording stopped for ID {self.rec_id}")
                break  # Add this line
            time.sleep(0.1)
            
    
    def delete_recordings(self, rec_id):
        remove_data(rec_id)
        print(f"Recording with ID {rec_id} has been deleted.")


class Autopilot:
    def __init__(self, movement_controller):
        log.debug("Initializing Autopilot")
        self.movement_controller = movement_controller
        self.car = self.movement_controller.get_car()
        self.controller = self.movement_controller.get_controller()
        log.debug("Car and controller initialized... Starting camera...")
        os.system("echo $USER | sudo -S systemctl restart nvargus-daemon")
        log.debug(f"Camera restarted...")
        self.camera = CSICamera(capture_device=0, width=512, height=512)
        self.cam_read = self.camera.read()
        self.camera.running = True
        self.autopilot_running = False
        
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(('localhost', 9999))

        log.debug("Starting controller thread...")
        # Start the thread to monitor the controller buttons
        self.autopilot_thread = threading.Thread(target=self.monitor_controller)
        self.autopilot_thread.daemon = True
        self.autopilot_thread.start()
        log.debug("Autopilot initialized successfully")

    def monitor_controller(self):
        while True:
            X_button = self.controller.buttons[2].value
            Y_button = self.controller.buttons[3].value

            if X_button == 1 and not self.autopilot_running:
                self.start_autopilot()
            if Y_button == 1 and self.autopilot_running:
                self.stop_autopilot()

            time.sleep(0.1)

    def start_autopilot(self):
        log.debug("Autopilot started")
        self.autopilot_running = True
        self.car.throttle = 0.2  # Set a constant speed

        while self.autopilot_running:
            frame = self.camera.value
            steering_angle = self.get_steering_angle_from_server(frame)
            log.debug(f"Predicted steering angle: {steering_angle}")
            self.car.steering = steering_angle
            time.sleep(0.1)

    def stop_autopilot(self):
        log.debug("Autopilot stopped")
        self.autopilot_running = False
        self.car.throttle = 0
        self.car.steering = 0

    def get_steering_angle_from_server(self, img):
        try:
            # Preprocess image
            img_data = img.tobytes()

            # Send the length of the image data
            self.client.sendall(len(img_data).to_bytes(4, 'big'))

            # Send the image data
            self.client.sendall(img_data)

            # Receive the steering angle
            steering_angle_data = self.client.recv(1024)
            steering_angle = float(steering_angle_data.decode('utf-8'))

            # Amplificate a bit
            if steering_angle >= 0.1 and steering_angle <= 0.7:
                steering_angle += 0.1
            elif steering_angle <= -0.1 and steering_angle >= -0.7:
                steering_angle -= 0.1

            return steering_angle
        except Exception as e:
            log.error(f"Error communicating with model server: {e}")
            return 0.0  # Default steering angle in case of error

    def show_camera(self):
        image0_widget = ipywidgets.Image(format='jpeg')

        image0_widget.value = bgr8_to_jpeg(self.cam_read)

        def update_image0(change):
            self.cam_read = change['new']
            image0_widget.value = bgr8_to_jpeg(self.cam_read)

        self.camera.observe(update_image0, names='value')

        return ipywidgets.HBox([image0_widget])