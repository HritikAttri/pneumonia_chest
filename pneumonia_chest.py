import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import itertools
from glob import glob
import cv2
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
%matplotlib inline

def get_data(data_dir):
    X = []
    y = []
    for label_dir in glob(data_dir + "/*"):
        if label_dir.split("/")[-1] == "NORMAL":
            label = 0
        else:
            label = 1
        for image_file in tqdm(glob(label_dir + "/*.jp*g")):
            img_file = cv2.imread(image_file)
            if img_file is not None:
                img_array = np.asarray(cv2.resize(img_file, (224, 224)))
                X.append(img_array)
                y.append(label)
        
    return np.asarray(X), np.asarray(y)

X_train, y_train = get_data("../input/chest-xray-pneumonia/chest_xray/train")
X_test, y_test = get_data("../input/chest-xray-pneumonia/chest_xray/test")

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# smt = SMOTE()

# X_train_new, y_train_new = smt.fit_sample(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))
# X_train_new = X_train_new.reshape(X_train_new.shape[0], 224, 224, 3)
# y_train_new = to_categorical(y_train_new, num_classes=2)

rus = RandomUnderSampler(sampling_strategy="auto")
X_train, y_train = rus.fit_sample(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))
X_train = X_train.reshape(X_train.shape[0], 224, 224, 3)
y_train = to_categorical(y_train, num_classes=2)

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

def keras_model(block_num=2):
    inp = Input(shape=(224, 224, 3))
    k = BatchNormalization()(inp)
    k = Conv2D(32, (7,7), padding="same",activation="relu",strides=(2,2))(k)
    k = MaxPool2D(pool_size=(3, 3), padding="same",strides=(2,2))(k) 
    k = Conv2D(32, (3,3), padding="same",activation="relu",strides=(1,1))(k)
    k = MaxPool2D(pool_size=(3, 3), padding="same",strides=(2,2))(k)
    for j in range(1,block_num+1):
        out_conv = []
        for i in [(1,1),(3,3),(5,5),(0,0)]:
            p = k
            if i == (1,1):
                p = Conv2D(32, (1,1), padding="same",activation="relu")(p)
                out_conv.append(Conv2D(32, (1,1), padding="same",activation="relu")(p))
            elif i == (0,0):
                p = MaxPool2D(pool_size=(2, 2), padding="same",strides=(1,1))(p)
                out_conv.append(Conv2D(32, (1,1), padding="same",activation="relu")(p))
            else:
                p = Conv2D(32, (1,1), padding="same",activation="relu")(p)
                p = Conv2D(32, i, padding="same",activation="relu")(p)
                out_conv.append(Conv2D(32, i, padding="same",activation="relu")(p))
        x = concatenate(out_conv, axis = -1)
        k = x
    x = MaxPool2D(pool_size=(7, 7), padding="same",strides=(2,2))(x)
    x = Flatten()(x)
    y = Dense(2,activation="softmax")(x)
    model = Model(inp, y)
    
    return model

model = keras_model(4)
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

model_ckpt_path = "./weights_pneumonia.hdf5"
checkpoint = ModelCheckpoint(model_ckpt_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
callbacks_list = [checkpoint]

train_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, 
                               vertical_flip=True, rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_gen = train_gen.flow(X_train, y_train, batch_size=16, shuffle=True)
test_gen = test_gen.flow(X_test, y_test, batch_size=16)

history = model.fit_generator(train_gen, validation_data=test_gen, epochs=20, steps_per_epoch=X_train.shape[0] // 16, 
                              validation_steps=X_test.shape[0] // 16, callbacks=callbacks_list, use_multiprocessing=False, verbose=1)

model2 = keras_model(4)
model2.load_weights("./weights_pneumonia.hdf5")
model2.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

print("Evaluation model1: ", model.evaluate(X_test, y_test))
print("Evaluation model2: ", model2.evaluate(X_test, y_test))