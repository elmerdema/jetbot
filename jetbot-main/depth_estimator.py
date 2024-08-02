import os
import cv2
import numpy as np
import time
import pickle
import requests

DPI=96
STREAM_URL= 'http://192.168.0.104:3001'
STREAM_ENDPOINT = '/video_feed'
ENGINE_ENDPOINT = '/engine'
ENGINE_START_AGAIN_AFTER_FRAMES = 30 # Number of frames to wait before starting the engine again
OBJECT_DETECTION_THRESHOLD = 5 # Minimum number of frames with object detection to stop the engine

def load_calibration_data(filename='camera_calibration/calibration.pkl'):
    with open(filename, 'rb') as f:
        calibration_data = pickle.load(f)
    return calibration_data


def process_frame(left, right, K = None, D = None, calibrate = False):
    
    if calibrate:
        h, w = left.shape[:2]
        
        # Get optimal new camera matrices and regions of interest
        DIM = (w, h)
        K, D = np.array(K), np.array(D)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

        # Undistort the left image
        image_left = cv2.remap(left, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Undistort the right image
        image_right = cv2.remap(right, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imshow('Undistorted Left', image_left)
        cv2.imshow('Undistorted Right', image_right)
        cv2.waitKey(1)
    else:
        kernel_size = 3
        image_left = cv2.GaussianBlur(left, (kernel_size,kernel_size), 1.5)
        image_right = cv2.GaussianBlur(right, (kernel_size, kernel_size), 1.5)
        

    # Stereo matching and disparity calculation
    window_size = 9    
    left_matcher = cv2.StereoSGBM_create(
        numDisparities=96,
        blockSize=7,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=16,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(2)

    disparity_left = np.int16(left_matcher.compute(image_left, image_right))
    disparity_right = np.int16(right_matcher.compute(image_right, image_left))

    wls_image = wls_filter.filter(disparity_left, image_left, None, disparity_right)
    wls_image = cv2.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    wls_image = np.uint8(wls_image)

    cv2.imshow('disparity', cv2.applyColorMap(wls_image, cv2.COLORMAP_JET))
    
    # Focus on the top half of the wls_image
    threshold = 170
    area_threshold = 1000
    h, w = wls_image.shape
    top_half = wls_image[0:h//2, :]
    _, max_val, _, _ = cv2.minMaxLoc(top_half)
    detected_object = False
    if max_val > threshold:
        mask = cv2.inRange(top_half, threshold, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(wls_image, (x, y), (x+w, y+h), (255, 255, 255), 2)
            area = cv2.contourArea(cnt)
            if area > area_threshold:
                print(f"Object detected with area: {area}")
                detected_object = True

            
    cv2.imshow('disparity', cv2.applyColorMap(wls_image, cv2.COLORMAP_JET))
    cv2.waitKey(1)

    if detected_object:
        return True

def process_stream(calibrate = False):
    K = None
    D = None
    if calibrate:
        try:
            calibration_data = load_calibration_data()
            K = calibration_data['K']
            D = calibration_data['D']
        except:
            print("Error loading calibration data. Please run camera_calibration/calibration.py first.")
            return

    cap = cv2.VideoCapture(STREAM_URL + STREAM_ENDPOINT)
    total_frames = 0
    start_time = time.time()
    
    frames_without_object_detection = 0
    object_detection_frames = 0
    engine_status = 'start'
    engine_url = STREAM_URL + ENGINE_ENDPOINT
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        left_image = frame[:, 0:frame.shape[1]//2]
        right_image = frame[:, frame.shape[1]//2:]

        if process_frame(left_image, right_image, K, D, calibrate):
            object_detection_frames += 1
            if object_detection_frames > OBJECT_DETECTION_THRESHOLD:
                frames_without_object_detection = 0
                object_detection_frames = 0
                
                if engine_status == 'start':
                    requests.post(engine_url, data = {'action': 'stop'})
                    
                engine_status = 'stop'
        else:
            object_detection_frames = 0
                
        frames_without_object_detection += 1
        if frames_without_object_detection > ENGINE_START_AGAIN_AFTER_FRAMES:
            if engine_status == 'stop':
                requests.post(engine_url, data = {'action': 'start'})
            frames_without_object_detection = 0
            engine_status = 'start'
            
        total_frames += 1
        print(f"FPS: {total_frames/(time.time()-start_time)}")
    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    process_stream(calibrate = False)
