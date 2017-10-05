import torch.utils.data as data_utils
import os
import torch
import torchvision
from torchvision import transforms
import random
from PIL import Image
import pandas as pd
import numpy as np


class WikiartDataset(data_utils.Dataset):
    def __init__(self, config):
        self.dataset_path = config.get('wikiart_path')
        self.images = config.get('images_path')
        self.num_samples = config.get('train_size')
        self.ids_list = list(range(1, self.num_samples+1))
        # random.shuffle(self.ids_list)

    def __getitem__(self, index):
        dataset = pd.read_csv(self.dataset_path, sep=',')
        row = dataset.iloc[index]
        image = Image.open(self.images + "/"+row['location'])
        image = image.resize((300, 300))
        image = np.array(image).astype(np.float32)
        label = row['class']
        sample = {'image': torch.from_numpy(image), 'class': label}
        return sample

    def __len__(self):
        return len(self.ids_list)

wiki = WikiartDataset(config={'wikiart_path': '../data/impressionists.csv',
                              'images_path': '/home/bhavika/wikiart/Impressionism',
                              'train_size': 500})

dataloader = data_utils.DataLoader(wiki, batch_size=4, shuffle=True, num_workers=2)

for i, sample in enumerate(dataloader):
    print (i, sample['class'], sample['image'].size())