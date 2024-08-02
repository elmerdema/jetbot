import os
import numpy as np
import cv2
import tensorflow as tf
import argparse

def preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

def predict_steering_angle(model, img):
    img_preprocessed = preprocess(img)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension
    steering_angle = float(model.predict(img_preprocessed))
    return steering_angle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering Angle Prediction')
    parser.add_argument('dir_path', type=str, help='Path to the directory containing the images')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Path to save the output video')
    args = parser.parse_args()

    model_path = 'model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")

    # Explicitly specify the custom_objects to ensure loss function is recognized
    custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}

    # Try loading the model using the updated API to handle version differences
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except ValueError as e:
        print(f"Error loading model: {e}")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(loss='mse', optimizer='adam')

    print("Model loaded successfully!")

    if not os.path.isdir(args.dir_path):
        raise NotADirectoryError(f"The path provided is not a directory: {args.dir_path}")

    image_paths = sorted([os.path.join(args.dir_path, img) for img in os.listdir(args.dir_path) if img.endswith(('jpg', 'jpeg', 'png'))])
    if not image_paths:
        raise ValueError(f"No images found in directory: {args.dir_path}")

    frame_width, frame_height = 640, 480  # Set frame dimensions
    output_path = args.output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue
        
        steering_angle = predict_steering_angle(model, img)
        print(f"Image: {img_path} | Predicted Steering Angle: {steering_angle:.2f}")

        # Overlay the steering angle on the image
        overlay_text = f"Steering Angle: {steering_angle:.2f}"
        cv2.putText(img, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Resize image to match video frame size if needed
        img_resized = cv2.resize(img, (frame_width, frame_height))
        
        # Write the frame to the video
        out.write(img_resized)

    out.release()
    print(f"Video saved to {output_path}")
