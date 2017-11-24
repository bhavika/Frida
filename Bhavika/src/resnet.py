import torch.utils.data as data_utils
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18


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
        image = image.resize((224, 224))
        image = np.array(image).astype(np.float32)
        label = row['class']

        sample = {'image': torch.from_numpy(image), 'class': label}
        return sample

    def __len__(self):
        return len(self.ids_list)


print("Loading training data....")

wiki_train = WikiartDataset(config={'wikiart_path': '../data/train_demo.csv',
                              'images_path': '/home/bhavika/wikiart/Impressionism',
                              'size': 20})

print("Loading test data....")
wiki_test = WikiartDataset(config={'wikiart_path': '../data/test_demo.csv',
                              'images_path': '/home/bhavika/wikiart/Impressionism',
                              'size': 18})

wiki_train_dataloader = data_utils.DataLoader(wiki_train, batch_size=1, shuffle=True, num_workers=2)
wiki_test_dataloader = data_utils.DataLoader(wiki_test, batch_size=1, shuffle=True, num_workers=2)


net = resnet18(pretrained=True)

# Defining the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


def get_classes(filepath):
    data = pd.read_csv(filepath, sep=',')
    return list(data[0:1000]['class'].unique())

classes = get_classes('../data/train_demo.csv')

# Train

for epoch in range(2):

    print("Epoch: ", epoch)

    running_loss = 0.0

    for i, data in enumerate(wiki_train_dataloader, 0):
        inputs, labels = data['image'], data['class']

        # make this 4, 32, 32, 3 -> 4, 3, 32, 32
        inputs = inputs.view(1, 3, 224, 224)

        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

print("Predicting on the test set... ")
class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
for data in wiki_test_dataloader:
    images, labels = data['image'], data['class']
    images = images.view(1, 3, 224, 224)
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(1):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



