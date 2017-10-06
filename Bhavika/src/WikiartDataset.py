import torch.utils.data as data_utils
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class WikiartDataset(data_utils.Dataset):
    def __init__(self, config):
        self.dataset_path = config.get('wikiart_path')
        self.images = config.get('images_path')
        self.num_samples = config.get('size')
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


wiki_train = WikiartDataset(config={'wikiart_path': '../data/impressionists.csv',
                              'images_path': '/home/bhavika/wikiart/Impressionism',
                              'size': 500})

wiki_test = WikiartDataset(config={'wikiart_path': '../data/impressionists.csv',
                              'images_path': '/home/bhavika/wikiart/Impressionism',
                              'size': 500})

wiki_train_dataloader = data_utils.DataLoader(wiki_train, batch_size=4, shuffle=True, num_workers=2)
wiki_test_dataloader = data_utils.DataLoader(wiki_test, batch_size=4, shuffle=True, num_workers=2)

net = Net()

# Defining the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def get_classes(filepath):
    data = pd.read_csv(filepath, sep=',')
    return list(data[0:1000]['class'].unique())

classes = get_classes('../data/impressionists.csv')
