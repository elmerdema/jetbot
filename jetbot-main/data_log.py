import os
from datetime import datetime
import cv2
import pandas as pd
import shutil


def save_data(left_cam_image, right_cam_image, steering_angle, rec_id=None):
    # Check if the steering angle is within the valid range
    if not -1 <= steering_angle <= 1:
        raise ValueError("Steering angle must be between -1 and 1")

    # If rec_id is not provided, use an integer value that is 1 more than the other folders if they exist
    if rec_id is None:
        rec_id = max([int(d) for d in os.listdir('jetson_data') if d.isdigit()], default=0) + 1

    # Create the main directory if it doesn't exist
    main_dir = os.path.join('jetson_data', str(rec_id))
    os.makedirs(main_dir, exist_ok=True)

    # Create the 'Left' and 'Right' directories
    left_dir = os.path.join(main_dir, 'Left')
    right_dir = os.path.join(main_dir, 'Right')
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    # Get the current timestamp with microseconds
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # Save the images with the timestamp as the name
    left_image_path = os.path.join(left_dir, f"{timestamp}.jpg")
    right_image_path = os.path.join(right_dir, f"{timestamp}.jpg")
    cv2.imwrite(left_image_path, left_cam_image)
    cv2.imwrite(right_image_path, right_cam_image)

    # Append the filenames and steering angle to log.csv
    # log.csv might not exist, so we need to check
    log_file = os.path.join(main_dir, 'log.csv')
    if not os.path.exists(log_file):
        pd.DataFrame(columns=['Left', 'Right', 'Steering']).to_csv(log_file, index=False)
        
    # Append the data to the log file
    pd.DataFrame([[left_image_path, right_image_path, steering_angle]], columns=['Left', 'Right', 'Steering'])\
        .to_csv(log_file, mode='a', header=False, index=False)

    print(f"Data saved successfully in {main_dir} directory.")


def remove_data(rec_id):
    dir_name = os.path.join('jetson_data', str(rec_id))
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
        print(f"Directory {dir_name} has been removed successfully.")
    else:
        print(f"Directory {dir_name} does not exist.")