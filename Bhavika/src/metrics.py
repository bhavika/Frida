import pickle
import sklearn.metrics
from constants import pickle_path

pkl_file_act = open(pickle_path+'rnet18_re_y_actual_1e-05_100.pkl', 'rb')
pkl_file_pred = open(pickle_path+'rnet18_re_ypred_1e-05_100.pkl', 'rb')

rnet18_re_act = pickle.load(pkl_file_act)
rnet18_re_pred = pickle.load(pkl_file_pred)

print(sklearn.metrics.accuracy_score(rnet18_re_pred, rnet18_re_act)*100)