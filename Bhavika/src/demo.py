import pickle
import torch
from constants import *
from cnn import Net
from resnet import resnet18
from resnet_retrained import _classifier
from wikiart import WikiartDataset
import torch.utils.data as data_utils
from torch.autograd import Variable


def load_model(path):
    checkpoint = torch.load(path)
    net = checkpoint['model']
    state = checkpoint['state_dict']
    return net, state


def predict(arch, trained_models):
    wiki_test = WikiartDataset(config={'wikiart_path': demo_data_csv, 'images_path': demo_data, 'size': 3,
                                       'arch': arch})

    wiki_test_dataloader = data_utils.DataLoader(wiki_test, batch_size=1, shuffle=True, num_workers=2,
                                                 drop_last=False)

    # load a model
    path = pickle_path + trained_models[arch]
    y_pred = []

    if arch == 'cnn':
        cnn_model = load_model(path)
        net = cnn_model[0]
        sz = 32
    elif arch=='resnet18':
        resnet_pre_model = load_model(path)
        net = resnet_pre_model[0]
        sz = 224
    elif arch == 'resnet18_re':
        resnet_tuned_model = load_model(path)
        net = resnet_tuned_model[0]
        sz = 224

    for data in wiki_test_dataloader:
        images, labels = data['image'], data['class']
        batchsize = images.shape[0]
        images = images.view(batchsize, 3, sz, sz)
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()

        y = predicted[0]
        y_pred.append(y)

    return y_pred


predictions = predict('cnn', trained_models=trained_models)
