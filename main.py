import os
import copy
import h5py
import pickle
from glob import glob
from random import shuffle

import cv2
from PIL import Image
from IPython.display import SVG

import numpy as np
import pandas as pd

import sklearn.utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, BatchNormalization, Convolution2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, concatenate, UpSampling2D, Conv2DTranspose

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import model_to_dot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_new_model(input_shape=(54, 54, 3), p=0.5, n_class=11, n_len=6):

    inputs = Input(((input_shape[0], input_shape[1], input_shape[2])))
    
    x = BatchNormalization()(inputs)
    x = Convolution2D(48, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(p/4)(x)
    
    x = BatchNormalization()(x)
    x = Convolution2D(64, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(p/4)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(128, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(p/2)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(160, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(p/2)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(p)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(p)(x)
    
    x = BatchNormalization()(x)
    x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(p)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(p)(x)
    
    x = Flatten()(x)
    x = Dense(3072, activation='relu')(x)
    x = Dense(3072, activation='relu')(x)

    l = Dense(n_len, activation='softmax')(x)
    c1 = Dense(n_class, activation='softmax')(x)
    c2 = Dense(n_class, activation='softmax')(x)
    c3 = Dense(n_class, activation='softmax')(x)
    c4 = Dense(n_class, activation='softmax')(x)
    c5 = Dense(n_class, activation='softmax')(x)
    
    output = [l, c1, c2, c3, c4, c5]
    
    model = Model(inputs=inputs, outputs=output)

    return model

def generate_crop(filepath, dataframe, expand_by=0.3, verbose=0, crop_sz=(64, 64), random_crop=True):

    # 1 - open the image and store img dimensions
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_h = img.shape[0]
    img_w = img.shape[1]
    
    if verbose>0:
        print( "img_h: ", img_h, "    img_w: ", img_w, "\n")

    # 2 - find bounding box for whole street number
    fn = filepath.split('/')[-1].split('.')[0]
#     print(fn)
    left = np.min(dataframe.loc[int(fn) - 1].left)
    top = np.min(dataframe.loc[int(fn) - 1].top)
    bottom = (np.max(dataframe.loc[int(fn) - 1].top) + np.max(dataframe.loc[int(fn) - 1].height))
    right = (np.max(dataframe.loc[int(fn) - 1].left) + np.max(dataframe.loc[int(fn) - 1].width))

    if verbose>0:
        print ("left: ", left, "    right: ", right, "    top: ", top, "    bottom: ", bottom, "\n")

    # 3 - Expand bounding box by X%
    mid_x = (left + right) // 2
    mid_y = (top + bottom) // 2
    new_h = np.abs(bottom - top) * (1 + expand_by)
    new_w = np.abs(right - left) * (1 + expand_by)

    if verbose>0:
        print( "mid_x: ", mid_x, "    mid_y: ", mid_y, "    new_h: ", new_h, "    new_w: ", new_w, "\n")
    
    # New points will be determined by the calculations above and the original image size
    left = np.max((0, mid_x - new_w // 2)).astype(np.uint)
    right = np.min((img_w, mid_x + new_w // 2)).astype(np.uint)
    top = np.max((0, mid_y - new_h // 2)).astype(np.uint)
    bottom = np.min((img_h, mid_y + new_h // 2)).astype(np.uint)

    if verbose>0:
        print( "n_left: ", left, "    n_right: ", right, "    n_top: ", top, "    n_bottom: ", bottom, "\n")
    
    # 4 - Crop image within bounding box
    cropped = img[top:bottom, left:right, :].copy()

    # 5 - Rescale to 64 x 64
    rescaled = cv2.resize(cropped, crop_sz)
    
    if random_crop:
        dx = np.random.randint(0, 10)
        dy = np.random.randint(0, 10)
        rescaled = rescaled[dx:dx+54, dy:dy+54, :]
    
    return rescaled

def get_label(filepath, dataframe, maxlength=5):
    
    fn = filepath.split('/')[-1].split('.')[0]

    label = dataframe.loc[int(fn) - 1].label

    l = np.zeros(maxlength+1)
    try:
        l[len(label)] = 1
    except:
        l[0] = 1

    y = np.zeros((5, 11), dtype=int)

    for i in range(5):
        try:
            y[i][int(label[i])] = 1
        except:
            y[i][0] = 1

    return [l, y[0], y[1], y[2], y[3], y[4]]

def new_generator(filepath_list, dataframe, batch_size=32, crop_sz=(64, 64),
                  shuffle_data=True, random_crop=True, return_labels=True):

    num_samples = len(filepath_list)
    filelist = copy.copy(filepath_list)

    if shuffle_data:
        shuffle(filelist)

    while True:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = filelist[offset:offset + batch_size]

            if shuffle_data:
                shuffle(batch_samples)

            images = []
            length = []
            digits = []

            for batch_sample in batch_samples:
                img = generate_crop(batch_sample, dataframe, crop_sz=crop_sz, random_crop=random_crop)
                y = np.zeros((5, 11), dtype='int')
                [l, y[0, :], y[1, :], y[2, :], y[3, :], y[4, :]] = get_label(batch_sample, dataframe)
            
                images.append(img)
                length.append(l)
                digits.append(y)

            X_train = np.array(images)
            if len(X_train.shape) == 3:
                X_train = np.expand_dims(X_train, -1)

            y_temp = np.array(digits)
            l = np.array(length)

            y1 = y_temp[:, 0, :]
            y2 = y_temp[:, 1, :]
            y3 = y_temp[:, 2, :]
            y4 = y_temp[:, 3, :]
            y5 = y_temp[:, 4, :]
            
            if return_labels:
                yield X_train, [l, y1, y2, y3, y4, y5]
            else:
                yield X_train

def new_convert_label(label):
    l = label[0]
    labels = label[1:]
    n_label = ""
    for digit in labels:
        if np.argmax(digit) == 0:
            n_digit = ""
        elif np.argmax(digit) == 10:
            n_digit = "0"
        else:
            n_digit = str(np.argmax(digit))
        n_label += n_digit
    return n_label


def new_convert_output(model_output):
    l = model_output[0]
    digits = np.array(model_output[1:]).swapaxes(0, 1)
    labels = []
    for i in range(len(l)):
        label = new_convert_label(([l[i]], digits[i, 0, :], digits[i, 1, :],
                                  digits[i, 2, :], digits[i, 3, :], digits[i, 4, :]))
        labels.append(label)

    return labels

new_model = get_new_model()
new_model.load_weights('models/new_model.10-1.14.hdf5')

def predict():
    print("PREDICTING")
    new_images = []
    new_filenames = []

    filelist = os.listdir('Extra/')

    for file in filelist:

        try:
            img = cv2.imread('Extra/' + file, cv2.IMREAD_COLOR)
        except:
            img = None

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (54, 54))
            new_images.append(img)
            new_filenames.append(file)
            
    new_images = np.array(new_images)
    test_new_preds = new_model.predict(new_images)
#    print(test_new_preds)
    new_pred_labels = new_convert_output(test_new_preds)
    
    rows_to_plot = 2
    cols_to_plot = 5

    f = plt.figure(figsize=(12, 18))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.8)
    for i, (pred, img) in enumerate(zip(new_pred_labels, new_images)):
        f.add_subplot(rows_to_plot, cols_to_plot, i+1)
        plt.title("Predicted: " + pred)
        plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def train():
    print("TRAINING")
    # path = ''
    dataset_path = 'data'
    train_filenames = glob(dataset_path + '/train/*.png')
    
    f = open('train_digitStruct.mat.pickle', 'rb')
    trn_data = pickle.load(f)
    trn_data = pd.DataFrame.from_dict(trn_data)
    
    
    trn_label_len = []
    for label in trn_data['label']:
        label_len = len(label)
        trn_label_len.append(label_len)
    
    
    # Create samples to test the generator
    i = 0
    for a, b in new_generator(train_filenames, trn_data, batch_size=25): 
        test_imgs = a
        test_lbls = b
        
        i += 1
        
        if i > 1:
            break
            
    trn_filenames, val_filenames = train_test_split(train_filenames, test_size=0.2)
    new_model.summary()
    
    new_model = get_new_model()
    optimizer = Adam(lr=1e-3)
    new_model.compile(optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    callbacks = [ModelCheckpoint('models/new_model.{epoch:02d}-{val_loss:.2f}.hdf5')]

    trn_generator = new_generator(trn_filenames, trn_data, batch_size=128)
    val_generator = new_generator(val_filenames, trn_data, batch_size=128)
  
    new_model.fit_generator(trn_generator,
                    epochs=10,
                    steps_per_epoch=200,
                    validation_data=val_generator,
                    validation_steps=200,
                    callbacks=callbacks,
                    verbose=1)

    teste_out = new_model.predict(test_imgs)
    
    output = new_convert_output(teste_out)

    rows_to_plot = 5
    cols_to_plot = 5

    f = plt.figure(figsize=(12, 6))

    for i, (pred, img) in enumerate(zip(output, test_imgs)):
        f.add_subplot(rows_to_plot, cols_to_plot, i+1)
        plt.title("Predicted: " + pred)
        plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    validate()     
    
def validate():
    print("VALIDATION")
    f = open('digitStruct.mat.pickle', 'rb')
    tst_data = pickle.load(f)
    tst_data = pd.DataFrame.from_dict(tst_data)

    tst_filenames = glob('data/test/*.png')
    
    tst_imgs = []
    tst_labels = []
    for filename in tst_filenames:
        img = generate_crop(filename, tst_data, crop_sz=(54, 54), random_crop=False)
        lbl = get_label(filename, tst_data)
        n_lbl = new_convert_label(lbl)
        tst_labels.append(n_lbl)
        tst_imgs.append(img)

    tst_imgs = np.array(tst_imgs)
    
    tst_pred = new_model.predict(tst_imgs, batch_size=128)
    
    pred_labels = new_convert_output(tst_pred)
    
    tst_accuracy = accuracy_score(tst_labels, pred_labels)
    print ("Accuracy: ", np.round(tst_accuracy*100, 1), "%")

def main():
    print("1. Train the model")
    print("2. Predict the digit")
    print("3. Validate the model")
    c = int(input())
    if c == 1:
        train()
    elif c == 2:
        predict()
    else:
        validate()

if __name__ =='__main__':
    main()
