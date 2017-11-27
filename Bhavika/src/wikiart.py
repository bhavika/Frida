import torch.utils.data as data_utils
import torch
import pandas as pd
from PIL import Image
import numpy as np


class WikiartDataset(data_utils.Dataset):
    def __init__(self, config):
        self.dataset_path = config.get('wikiart_path')
        self.images = config.get('images_path')
        self.num_samples = config.get('size')
        self.ids_list = list(range(1, self.num_samples+1))
        self.arch = config.get('arch')
        # random.shuffle(self.ids_list)

    def __getitem__(self, index):
        dataset = pd.read_csv(self.dataset_path, sep=',')
        row = dataset.iloc[index]
        image = Image.open(self.images + "/"+row['location'])
        if self.arch == 'cnn':
            image = image.resize((32, 32))
        else:
            image = image.resize((224, 224))
        image = np.array(image).astype(np.float32)
        label = row['class']

        sample = {'image': torch.from_numpy(image), 'class': label}
        return sample

    def __len__(self):
        return len(self.ids_list)