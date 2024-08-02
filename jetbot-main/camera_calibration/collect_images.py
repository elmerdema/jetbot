import cv2
import os

# Use the same VIDEO_STREAM_URL as in the stereo vision script
VIDEO_STREAM_URL = 'http://192.168.0.104:3001/video_feed'

# Determine the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, 'images')

# Ensure the 'images' directory exists
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

cap = cv2.VideoCapture(VIDEO_STREAM_URL)

if not cap.isOpened():
    print("Error: Unable to open video stream")
    exit()

num = 0

while cap.isOpened():
    success, img = cap.read()

    if not success:
        print("Failed to capture image")
        break

    cv2.imshow('Img', img)

    k = cv2.waitKey(5)
    
    if k == 27:  # ESC key to exit
        break
    elif k == ord('s'):  # 's' key to save the image
        image_path = os.path.join(image_dir, f'img{num}.png')
        # Write only half the image since we are using a stereo camera
        img = img[:, 0:img.shape[1]//2]
        cv2.imwrite(image_path, img)
        
        if os.path.exists(image_path):
            print(f"Image {num} saved to {image_path}")
        else:
            print(f"Failed to save image {num}")
        
        num += 1

cap.release()
cv2.destroyAllWindows()
