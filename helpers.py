import cv2
from PIL import Image
import numpy as np


def read_img(path):
    """
    Returns a numpy array image from path
    """
    img = cv2.imread(path.strip())
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_img(path):
    image = Image.open(path)
    image = image.convert('RGB')
    return image


def preprocess_data(data):
    """
    Return processed data
    """
    return data['center'], data['steering']


def images_to_nparray(image_list, w=160, h=320):
    output = []
    for image in image_list:
        img = np.array(list(image.getdata()), dtype='uint8')
        img = np.reshape(img, (w, h, 3))
        output.append(img)

    return np.array(output, dtype='uint8')


def get_batch(data, batch_size=24*6):
    batch = data.sample(int(batch_size/6))
    img_paths = np.concatenate((batch['center'].str.strip().values,
                               batch['center_flipped'].str.strip().values,
                               batch['left'].str.strip().values,
                               batch['left_flipped'].str.strip().values,
                               batch['right'].str.strip().values,
                               batch['right_flipped'].str.strip().values))
    steerings = np.concatenate((batch['center_steering'].values,
                               batch['center_steering_flip'].values,
                               batch['left_steering'].values,
                               batch['left_steering_flip'].values,
                               batch['right_steering'].values,
                               batch['right_steering_flip'].values))
    img_list = []
    for path in img_paths:
        img = load_img(path)
        img = np.asarray(img, dtype='float32')
        img_list.append(img)

    imgs = np.array(img_list, dtype='float32')
    strs = np.array(steerings, dtype='float32')

    return imgs, strs


def train_sample_generator(train_df, batch_size=24*6):
    """
    Generate a batch of training data
    """
    while True:
        yield get_batch(train_df, batch_size=batch_size)


def valid_sample_generator(valid_df, batch_size_valid=24*6):
    return train_sample_generator(valid_df, batch_size_valid)


def crop_imgs(imgs, top=60, bottom=140):
    """
    Returns croppped image tensor
    """
    return imgs[:,top:bottom,:,:]

