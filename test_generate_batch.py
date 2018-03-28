import numpy as np
import os
# from data_preprocessing import get_stopwords
# from data_preprocessing import build_dataset
# from word2vec import read_data
# from word2vec import TRAIN_DIR
from data_preprocessing import build_dataset
from data_preprocessing import generate_batch

VOCABULARY_SIZE = 50000
BATCH_SIZE = 128*128*64*128 #10*2*2*2*32
WINDOW_SIZE = 6

print('TEST: generate_batch')

def read_data(directory, domain_words=-1):
    data = []
    # for domain in os.listdir(directory):
    # #for dirpath, dnames, fnames in os.walk(directory):
    limit = domain_words
    #     for f in os.listdir(os.path.join(directory, domain)):
    #         if f.endswith("6582.txt"):
    with open('dataset/DATA/TRAIN/HERALDRY_HONORS_AND_VEXILLOLOGY/47809.txt') as file:
        # sentences = []
        for line in file.readlines():
            split = line.lower().strip().split()
            if limit > 0 and limit - len(split) < 0:
                split = split[:limit]
            else:
                limit -= len(split)
            if limit >= 0 or limit == -1:
                data.append(split)
                # print(sentences)
        # data += sentences
        # data.append(sentences)
    return data

raw_data = read_data("dataset/DATA/TRAIN", domain_words=1000)

# print(raw_data)

data, dictionary, reverse_dictionary = build_dataset(raw_data, VOCABULARY_SIZE)


print("DATA:")
print(data)
batch_inputs, batch_labels = generate_batch(BATCH_SIZE, 0, 0, 0, WINDOW_SIZE, data)

for x, y in zip(batch_inputs, batch_labels):
    print("{},{}".format(x,y[0]))
