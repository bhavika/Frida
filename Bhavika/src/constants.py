bs = 4 # batch size

train_path = '../data/train_top15.csv'
test_path = '../data/test_top15.csv'

dataset_path = '../data/top15.csv'

images_path = '/home/bhavika/wikiart/Impressionism'
n = 15*40
train_size = 360
test_size = 240

mappings_path = '../data/top15_mappings.csv'

pickle_path = '../models/'


trained_models = {'cnn': 'CNN_checkpoint.pth.tar', 'resnet18': 'resnet18_checkpoint.pth.tar',
                  'resnet18_re': 'resnet18_re_checkpoint.pth.tar'}

demo_data = '../data/demo/'
train_demo_data = '../data/TrainDemo/'

demo_data_csv = '../data/demo/demo.csv'