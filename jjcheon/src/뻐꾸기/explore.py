import pandas as pd
from PIL import Image

base_address = '/home/bhavika/wikiart/'


def read_artist_data(base_address, filepath):
    filepath = base_address + filepath
    paintings_by_artist = pd.read_csv(filepath,names=['location', 'class'], header=0)
    paintings_by_artist['absolute_location'] = base_address+paintings_by_artist['location']
    collage = Image.new('RGB', (800, 800))
    for i in range(paintings_by_artist.shape[0]):
        link = paintings_by_artist.iloc[i]['absolute_location']
        print(link)
    # This will only open the last image - only to test.
    img = Image.open(link)
    img.show()

if __name__ == '__main__':
    train_file = '/artist_train.csv'
    read_artist_data(base_address=base_address,filepath=train_file)