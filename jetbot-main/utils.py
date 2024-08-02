import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import image as mpimg
from sklearn.utils import shuffle
import random
from imgaug import augmenters as iaa

# Function tested and checked
def getName(filePath):
    myImagePathL = filePath.split('/')[-3:]
    myImagePath = os.path.join(myImagePathL[0], myImagePathL[1], myImagePathL[2])
    return myImagePath

def import_data_info(path):
    columns = ['Left', 'Right', 'Steering']
    data = pd.DataFrame()
    for i, directory in enumerate(os.listdir(path)):
        current_path = os.path.join(path, directory, 'Left')
        csv_path = os.path.join(path, directory, 'log.csv')
        if os.path.isdir(current_path) and os.path.exists(csv_path):
            temp_df = pd.read_csv(csv_path, names=columns, header=0)
            temp_df['Left'] = temp_df['Left'].apply(lambda x: x.strip())
            temp_df['Steering'] = pd.to_numeric(temp_df['Steering'], errors='coerce')
            temp_df['Left'] = temp_df['Left'].apply(lambda x: os.path.join(x))
            temp_df = temp_df.iloc[20:-10]
            data = pd.concat([data, temp_df[['Left', 'Steering']].rename(columns={'Left': 'Image'})], ignore_index=True)
    data.dropna(inplace=True)  # Drop any rows with NaN values
    return data


def balance_data(data, display=True):
    n_bins = 20
    samples_per_bin = 1000
    hist, bins = np.histogram(data['Steering'], n_bins)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samples_per_bin, samples_per_bin))
        plt.title('Original Data')
        plt.show()
    remove_list = []
    for j in range(n_bins):
        list_ = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                if data['Steering'][i] == 0:
                    # This part randomly removes X% of the samples with a steering angle 0
                    if np.random.rand() < 0.2:
                        remove_list.append(i)
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)
    data.drop(data.index[remove_list], inplace=True)
    data.reset_index(drop=True, inplace=True)  # Reset the index

    # Augment data by flipping images and reversing steering angles
    augmented_data = pd.DataFrame(columns=data.columns)  # Create a new DataFrame for augmented data
    for i in range(len(data['Steering'])):
        if data['Steering'][i] > -1 and data['Steering'][i] < 0 or data['Steering'][i] > 0 and data['Steering'][i] < 1:
            img = cv2.imread(data['Image'][i])
            if img is not None:
                flipped_image_path = data['Image'][i].replace('.jpg', '_flipped.jpg')  # Create a new path for the flipped image
                cv2.imwrite(flipped_image_path, img)  # Save the flipped image
                reversed_angle = -data['Steering'][i]
                new_row = pd.DataFrame({'Image': [flipped_image_path], 'Steering': [reversed_angle]})
                augmented_data = pd.concat([augmented_data, new_row], ignore_index=True)
            else:
                print(f"Error loading image: {data['Image'][i]}")
                continue

    data = pd.concat([data, augmented_data])  # Concatenate original and augmented data

    # Plot the data after balancing
    if display:
        hist, _ = np.histogram(data['Steering'], n_bins)
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samples_per_bin, samples_per_bin))
        plt.title('Balanced Data')
        plt.show()

    return data

def preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

def augmentImage(img_path,steering):
    print('Augmenting Image')
    img = mpimg.imread(img_path)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(66, 200, 3)),
        tf.keras.layers.Conv2D(24, (5,5), strides=(2,2), activation='relu'),
        tf.keras.layers.Conv2D(36, (5,5), strides=(2,2), activation='relu'),
        tf.keras.layers.Conv2D(48, (5,5), strides=(2,2), activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

def loadData(path, data):
    imagePath = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        imagePath.append(indexed_data[0])
        steering.append(float(indexed_data[1]))
    imagePath = np.asarray(imagePath)
    steering = np.asarray(steering)
    return imagePath, steering


def dataGen(imagePaths, steeringAngles, batchSize, trainFlag):
    while True:
        batch_img = []
        batch_steering = []

        for i in range(batchSize):
            while True:
                random_index = random.randint(0, len(imagePaths) - 1)
                if trainFlag:
                    try:
                        img, steering = augment_image(imagePaths[random_index], steeringAngles[random_index])
                    except Exception as e:
                        continue  # Skip this image if there's an error
                else:
                    img = mpimg.imread(imagePaths[random_index])
                    steering = steeringAngles[random_index]
                    if img is None:
                        continue  # Skip if the image is not loaded properly

                if img is not None:
                    img = preprocess(img)
                    batch_img.append(img)
                    batch_steering.append(steering)
                    break  # Only break if image is successfully loaded and processed

        yield (np.asarray(batch_img), np.asarray(batch_steering))
