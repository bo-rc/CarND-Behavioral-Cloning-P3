# Load pickled data
import pickle
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import Sequential
from keras.backend import tf as ktf
from keras.optimizers import Adam
from helpers import *
from keras_multiGPU import make_parallel

csv_filename = 'driving_log_aug.csv'
data_path = "../sim_data/data/"

data_csv_path = data_path + csv_filename

data_csv_df = pd.read_csv(data_csv_path, index_col=False)

data = data_csv_df.sample(n=len(data_csv_df))

split_train = int(0.75*len(data))
split_valid = int(0.95*len(data))
split_test = len(data)

data_train = data[:split_train]
data_valid = data[split_train:split_valid]
data_test = data[split_valid:split_test]


def nvidia_net(input_shape=(160, 320, 3)):
    model = Sequential()

    # 160x320x3 -> 80x320x3
    model.add(Lambda(crop_imgs, input_shape=input_shape))

    # 80x320x3 -> 80x320x3
    model.add(Lambda(lambda x: x / 255. - 0.5))

    # -> 38x158x24
    model.add(Conv2D(24, kernel_size=(5, 5),
                     strides=(2, 2),
                     activation='relu'))

    # -> 17x77x36
    model.add(Conv2D(36, kernel_size=(5, 5),
                     strides=(2, 2),
                     activation="relu"))

    # -> 7x37x48
    model.add(Conv2D(48, kernel_size=(5, 5),
                     strides=(2, 2),
                     activation="relu"))

    # -> 3x18x64
    model.add(Conv2D(64, kernel_size=(3, 3),
                     strides=(2, 2),
                     activation="relu"))

    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="relu"))

    model.add(Dense(1))

    make_parallel(model, 2)
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model


BATCH_SIZE = 24*6
BATCH_SIZE_VALID = 12*6
EPOCHS = 5

DATA_PATH = "../sim_data/data/"

model_nv = nvidia_net((160,320,3))

print(model_nv.summary())

n_data_train = len(data_train)
n_steps = int(n_data_train/BATCH_SIZE)

n_data_valid = len(data_valid)
n_steps_valid = int(n_data_valid/BATCH_SIZE_VALID)

values = model_nv.fit_generator(train_sample_generator(data_train, BATCH_SIZE),
                             validation_data=valid_sample_generator(data_valid, BATCH_SIZE_VALID),
                             steps_per_epoch=n_steps,
                             validation_steps=n_steps_valid,
                             epochs=EPOCHS)

model_nv.save('nvidia_net_wAugData.h5')

