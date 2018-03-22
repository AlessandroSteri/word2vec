import numpy as np
# from data_preprocessing import get_stopwords
# from data_preprocessing import build_dataset
from word2vec import read_data
# from word2vec import TRAIN_DIR
from data_preprocessing import build_dataset
from data_preprocessing import generate_batch

VOCABULARY_SIZE = 50000

print('TEST')

raw_data = read_data("dataset/DATA/TRAIN", domain_words=1)

# WORDS = ['the', 'barometric', 'formula,', 'sometimes', 'called', 'the',
#          'exponential', 'atmosphere', 'or', 'isothermal', 'atmosphere,', 'is',
#          'a', 'formula', 'used', 'to', 'model', 'how', 'the', 'pressure', '(or',
#          'density)', 'of', 'the', 'air', 'changes', 'with', 'altitude.', 'there',
#          'are', 'two', 'different', 'equations', 'for', 'computing', 'pressure',
#          'at', 'various', 'height', 'regimes', 'below', '86', 'km', '(or',
#          '278,400', 'feet).', 'the', 'first', 'equation', 'is', 'used', 'when',
#          'the', 'value', 'of', 'standard', 'temperature', 'lapse', 'rate', 'is',
#          'not', 'equal', 'to', 'zero;', 'the', 'second', 'equation', 'is',
#          'used', 'whenl', 'standard', 'temperature', 'lapse', 'rate', 'equals',
#          'zero.']
#

data, dictionary, reverse_dictionary = build_dataset(raw_data, VOCABULARY_SIZE)

# DATA, dictionary, reverse_dictionary = build_dataset(WORDS, VOCABULARY_SIZE)
#
#
# print(WORDS[10])
# print(WORDS[15])
# print(len(WORDS))
# for w in WORDS[10:15]:
#     print('hi' + w)

DATA = [1, 2, 3, 4, 5, 6, 7, 8, 9]
x_train, y_train = generate_batch(10, 1, 2, DATA)

print("DATA:")
print(DATA)
print("x_train:")
print(x_train)
print("y_train:")
print(y_train)

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print('x_train shape', x_train.shape, '\n'
      'y_train_shape', y_train.shape)




