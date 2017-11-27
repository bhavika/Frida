import os
import pandas as pd
from constants import *
import shutil

# we are using the train path here because those are the only images we trained on & interested in evaluating
# in relation to the test images during the demo
dataset = pd.read_csv(train_path, sep=',')


def copy_paintings(path):
    os.mkdir(path)
    for i in range(dataset.shape[0]):
        row = dataset.iloc[i]
        img_path = images_path + "/" + row['location']
        shutil.copy2(img_path, path)


copy_paintings('../data/TrainDemo')