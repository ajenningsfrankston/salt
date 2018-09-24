import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


from sklearn.model_selection import train_test_split

from tqdm import tqdm

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

img_size_ori = 101
img_size_target = 128

plt.style.use('seaborn-white')

sns.set_style("white")


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    # res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    # res[:img_size_ori, :img_size_ori] = img
    # return res


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]


data_home = "../../salt_data"

train_df = pd.read_csv(data_home + "/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv(data_home + "/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

print(train_df.head())


train_df["images"] = [np.array(load_img(data_home + "/train/images/{}.png".format(idx), color_mode="grayscale")) / 255
                                                                                        for idx in tqdm(train_df.index)]