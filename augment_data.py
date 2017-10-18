import pandas as pd
from helpers import *

TEST = False

csv_filename = 'driving_log.csv'
data_path = "../sim_data/data/"
data_test_path = "../sim_data/test_data/"

if TEST:
    data_csv_path = data_test_path + csv_filename
else:
    data_csv_path = data_path + csv_filename

data_csv_df = pd.read_csv(data_csv_path, index_col=False, header=None)
data_csv_df.columns = ['center', 'left', 'right', 'center_steering', 'throttle', 'brake', 'speed']

# randomly drop 0-steering data
data_csv_df = data_csv_df.drop(data_csv_df.query('-0.0005 < center_steering < 0.0005').sample(frac=.7).index)

# correction magnitude for off-center steering
correction = 1.5 * data_csv_df['center_steering'].std()
print("correction steering value = ", correction)

data_csv_df['left_steering'] = data_csv_df['center_steering'] + correction
data_csv_df['right_steering'] = data_csv_df['center_steering'] - correction

# flip images horizontally, add to dataframe
data_csv_df['center_steering_flip'] = -data_csv_df['center_steering']
data_csv_df['left_steering_flip'] = -data_csv_df['left_steering']
data_csv_df['right_steering_flip'] = -data_csv_df['right_steering']

print("generating flipped data for center...")
filelist_center_img_flip = []
for filename in data_csv_df['center']:
    img = Image.open(filename.strip()).transpose(Image.FLIP_LEFT_RIGHT)
    flip_filename = filename.strip().rstrip('.jpg') + 'flipped.jpg'
    img.save(flip_filename)
    filelist_center_img_flip.append(flip_filename)

data_csv_df['center_flipped'] = pd.Series(filelist_center_img_flip).values

print("generating flipped data for left...")
filelist_left_img_flip = []
for filename in data_csv_df['left']:
    img = Image.open(filename.strip()).transpose(Image.FLIP_LEFT_RIGHT)
    flip_filename = filename.strip().rstrip('.jpg') + 'flipped.jpg'
    img.save(flip_filename)
    filelist_left_img_flip.append(flip_filename)

data_csv_df['left_flipped'] = pd.Series(filelist_left_img_flip).values

print("generating flipped data for right...")
filelist_right_img_flip = []
for filename in data_csv_df['right']:
    img = Image.open(filename.strip()).transpose(Image.FLIP_LEFT_RIGHT)
    flip_filename = filename.strip().rstrip('.jpg') + 'flipped.jpg'
    img.save(flip_filename)
    filelist_right_img_flip.append(flip_filename)

data_csv_df['right_flipped'] = pd.Series(filelist_right_img_flip).values

save_filename = data_csv_path.rstrip('.csv') + '_aug.csv'
data_csv_df.to_csv(save_filename)
