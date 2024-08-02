import socket
import numpy as np
import cv2
import tensorflow as tf
import logging
import threading
import os

# Set up logging
log = logging.getLogger("model_server")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('model_server.log')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

graph = tf.Graph()

def preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

def predict_steering_angle(model, img):
    img_preprocessed = preprocess(img)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension
    steering_angle = float(model.predict(img_preprocessed))
    return steering_angle

def handle_client(client_socket):
    custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
    try:
        model = tf.keras.models.load_model("model.h5", custom_objects=custom_objects)
    except ValueError as e:
        log.error(f"Error loading model: {e}")
        model = tf.keras.models.load_model("model.h5", compile=False)
        model.compile(loss='mse', optimizer='adam')
        
    # Run a test prediction to make sure the model is loaded correctly
    test_img = np.zeros((66, 200, 3), dtype=np.uint8)
    test_steering_angle = predict_steering_angle(model, test_img)
    log.debug(f"Model is running")

    try:
        while True:
            # Receive the length of the image data
            data_length = client_socket.recv(4)
            if not data_length:
                break
            data_length = int.from_bytes(data_length, 'big')

            # Receive the image data
            img_data = b""
            while len(img_data) < data_length:
                packet = client_socket.recv(data_length - len(img_data))
                if not packet:
                    break
                img_data += packet

            if len(img_data) != data_length:
                break

            # Convert image data to numpy array
            img = np.frombuffer(img_data, dtype=np.uint8)
            img = img.reshape((512, 512, 3))
            
            # Use cv2 read to convert the image to the correct format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            log.debug(f"Received image with shape: {img.shape}")

            # Predict steering angle
            steering_angle = predict_steering_angle(model, img)
            steering_angle_data = str(steering_angle).encode('utf-8')
            log.debug(f"Predicted steering angle: {steering_angle}")
            # Send back the steering angle
            client_socket.sendall(steering_angle_data)
    except Exception as e:
        log.error(f"Error handling client: {e}")

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 9999))
    server.listen(5)
    log.debug("Server listening on port 9999")

    while True:
        client_socket, addr = server.accept()
        log.debug(f"Accepted connection from {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

if __name__ == "__main__":
    main()
