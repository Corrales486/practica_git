import json
import os

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'
]
twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True,
                                  random_state=42)
twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories,
                                 shuffle=True,
                                 random_state=42)


def store_text_bunch(bunch, filename):
    with open(filename, 'w') as fout:
        json.dump([{
            'text': text,
            'target': int(target)
        } for text, target in zip(bunch.data, bunch.target)],
                  fout,
                  indent=1)


if __name__ == '__main__':
    base_data_dir = 'data/raw'
    os.makedirs(base_data_dir, exist_ok=True)

    store_text_bunch(twenty_train,
                     os.path.join(base_data_dir, 'train_data.json'))
    store_text_bunch(twenty_test, os.path.join(base_data_dir,
                                               'test_data.json'))
