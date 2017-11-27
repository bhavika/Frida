import pickle
import torch
from constants import *
from cnn import Net
from resnet import resnet18
from resnet_retrained import _classifier


def load_model(path):
    checkpoint = torch.load(path)
    net = checkpoint['model']
    state = checkpoint['state_dict']
    return net, state


resnet_pre_model = load_model(pickle_path+resnet18_pre)
resnet_tuned_model = load_model(pickle_path+resnet18_tuned)
cnn_model = load_model(pickle_path+cnn_model)

