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
import itertools
from constants import *
import pickle
from wikiart import WikiartDataset


def get_classes(filepath):
    data = pd.read_csv(filepath, sep=',')
    return list(data['artist'].unique()), list(data['class'].unique())

artists, classes = get_classes(mappings_path)


def main(learning_rate, epochs=100):
    print("Loading training data....")

    wiki_train = WikiartDataset(config={'wikiart_path': train_path, 'images_path': images_path, 'size': train_size,
                                        'arch':'resnet'})

    print("Loading test data....")
    wiki_test = WikiartDataset(config={'wikiart_path': test_path, 'images_path': images_path, 'size': test_size,
                                       'arch': 'resnet'})

    wiki_train_dataloader = data_utils.DataLoader(wiki_train, batch_size=bs, shuffle=True, num_workers=2,
                                                  drop_last=False)
    wiki_test_dataloader = data_utils.DataLoader(wiki_test, batch_size=bs, shuffle=True, num_workers=2,
                                                 drop_last=False)

    net = resnet18(pretrained=True)

    # Defining the loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # Train

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        running_loss = 0.0

        for i, data in enumerate(wiki_train_dataloader, 0):
            inputs, labels = data['image'], data['class']
            batchsize = inputs.shape[0]
            # make this 4, 32, 32, 3 -> 4, 3, 32, 32
            inputs = inputs.view(batchsize, 3, 224, 224)

            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            save_checkpoint({'epoch': epoch + 1, 'arch': 'resnet18', 'state_dict': net.state_dict(), 'model': net})

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    print("Predicting on the test set... ")
    class_correct = list(0. for i in range(15))
    class_total = list(0. for i in range(15))

    y_pred = []
    y_actual = []

    for data in wiki_test_dataloader:
        images, labels = data['image'], data['class']
        batchsize = images.shape[0]
        images = images.view(batchsize, 3, 224, 224)
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()

        for i in range(batchsize):
            label = labels[0]
            y_actual.append(label)
            class_correct[label] += c[i]
            y_pred.append(c[i])
            class_total[label] += 1

    pkl1_name = "resnet_ypred_{}_{}.pkl".format(learning_rate, epochs)
    pkl2_name = "resnet_y_actual_{}_{}.pkl".format(learning_rate, epochs)

    with open(pickle_path + pkl1_name, 'wb') as f:
        pickle.dump(y_pred, f)

    with open(pickle_path + pkl2_name, 'wb') as f2:
        pickle.dump(y_actual, f2)

    print("Correct classes", class_correct)
    print("Total count for each class", class_total)
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            artists[i], 100 * class_correct[i] / class_total[i]))


def save_checkpoint(state, path=pickle_path, filename='resnet18_checkpoint.pth.tar'):
    torch.save(state, path+filename)


def load_model(path):
    checkpoint = torch.load(path)
    net = checkpoint['model']
    state = checkpoint['state_dict']
    return net, state

if __name__ == '__main__':

    lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    epochs = [100]

    combinations = list(itertools.product(lrs, epochs))

    for c in combinations:
        print("Learning rate {}, no of epochs {}".format(c[0], c[1]))
        main(learning_rate=c[0], epochs=c[1])


