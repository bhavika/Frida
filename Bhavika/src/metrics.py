import pickle
import sklearn.metrics
from constants import pickle_path
import os

lrs = ['1e-05', '0.0001', '0.001', '0.01', '0.1']
epochs = '100'

for lr in lrs:
    for f in os.listdir(pickle_path):
        if f.endswith('.pkl'):
                if lr in f and epochs in f:
                    if 'rnet18' in f:
                        y_pred_path = 'rnet18_re_ypred_{}_{}.pkl'.format(lr, epochs)
                        y_act_path = 'rnet18_re_y_actual_{}_{}.pkl'.format(lr, epochs)
                        pkl_act = pickle.load(open(pickle_path+y_act_path, 'rb'))
                        pkl_pred = pickle.load(open(pickle_path+y_pred_path, 'rb'))
                        print("Accuracy for ResNet-18 (finetuned) with learning rate {}".format(lr))
                        print(sklearn.metrics.accuracy_score(pkl_pred, pkl_act)*100)
                        print("Precision for ResNet-18 (finetuned) with learning rate {}".format(lr))
                        print(sklearn.metrics.precision_score(pkl_pred, pkl_act, average='weighted') * 100)
                        print("Recall for ResNet-18 (finetuned) with learning rate {}".format(lr))
                        print(sklearn.metrics.recall_score(pkl_pred, pkl_act, average='weighted') * 100)
                    elif 'cnn' in f:
                        y_pred_path = 'cnn_ypred_{}_{}.pkl'.format(lr, epochs)
                        y_act_path = 'cnn_y_actual_{}_{}.pkl'.format(lr, epochs)
                        pkl_act = pickle.load(open(pickle_path + y_act_path, 'rb'))
                        pkl_pred = pickle.load(open(pickle_path + y_pred_path, 'rb'))
                        print("Accuracy for CNN - 2 layer with learning rate {}".format(lr))
                        print(sklearn.metrics.accuracy_score(pkl_pred, pkl_act) * 100)
                        print("Precision for CNN - 2 Layer with learning rate {}".format(lr))
                        print(sklearn.metrics.precision_score(pkl_pred, pkl_act, average='weighted') * 100)
                        print("Recall for CNN - 2 Layer with learning rate {}".format(lr))
                        print(sklearn.metrics.recall_score(pkl_pred, pkl_act, average='weighted') * 100)
                    elif 'resnet' in f:
                        y_pred_path = 'resnet_ypred_{}_{}.pkl'.format(lr, epochs)
                        y_act_path = 'resnet_y_actual_{}_{}.pkl'.format(lr, epochs)
                        pkl_act = pickle.load(open(pickle_path + y_act_path, 'rb'))
                        pkl_pred = pickle.load(open(pickle_path + y_pred_path, 'rb'))
                        print("Accuracy for ResNet (transfer learning) with learning rate {}".format(lr))
                        print(sklearn.metrics.accuracy_score(pkl_pred, pkl_act) * 100)
                        print("Precision for ResNet-18 (transfer learning) with learning rate {}".format(lr))
                        print(sklearn.metrics.precision_score(pkl_pred, pkl_act, average='weighted') * 100)
                        print("Recall for ResNet-18 (transfer learning) with learning rate {}".format(lr))
                        print(sklearn.metrics.recall_score(pkl_pred, pkl_act, average='weighted') * 100)
