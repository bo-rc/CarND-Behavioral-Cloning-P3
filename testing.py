import pandas as pd
from helpers import *

csv_filename = 'driving_log_aug.csv'
data_path = "../sim_data/data/"

data_csv_path = data_path + csv_filename

data_csv_df = pd.read_csv(data_csv_path, index_col=False)

data = data_csv_df.sample(n=len(data_csv_df))
print(list(data))

imgs, steerings = get_batch(data, 6)

print(imgs.shape)
print(steerings.shape)

