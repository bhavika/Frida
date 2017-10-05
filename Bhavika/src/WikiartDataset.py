from torch.utils.data import dataset
import os
import random
from PIL import Image
import pandas as pd
import numpy as np


class WikiartDataset(dataset.Dataset):
    def __init__(self, config):
        self.dataset_path = config.get('wikiart_path')
        self.images = config.get('images_path')
        self.num_samples = config.get('train_size')
        self.ids_list = list(range(1, self.num_samples+1))
        # random.shuffle(self.ids_list)

    def __getitem__(self, index):
        dataset = pd.read_csv(self.dataset_path, sep=',')
        row = dataset.iloc[index]
        image = Image.open(self.images+ "/"+row['location'])
        image = np.array(image)
        image = np.array(image).astype(np.float32)
        label = np.array(row['class']).astype(np.int)
        return image, label

    def __len__(self):
        return len(self.ids_list)

wiki = WikiartDataset(config={'wikiart_path': '../data/impressionists.csv',
                              'images_path': '/home/bhavika/wikiart/Impressionism',
                              'train_size': 500})

for i in range(wiki.__len__()):
    image, label = wiki.__getitem__(i)
    print (image, label)