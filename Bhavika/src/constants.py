bs = 4 # batch size

train_path = '../data/train_top15.csv'
test_path = '../data/test_top15.csv'

images_path = '/home/bhavika/wikiart/Impressionism'
n = 15*40
train_size = 360
test_size = 240

mappings_path = '../data/top15_mappings.csv'

pickle_path = '../models/'

cnn_model = 'CNN_checkpoint.pth.tar'
resnet18_pre = 'resnet18_checkpoint.pth.tar'
resnet18_tuned = 'resnet18_re_checkpoint.pth.tar'
