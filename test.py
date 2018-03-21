# from data_preprocessing import get_stopwords
# from data_preprocessing import build_dataset
# from word2vec import read_data
# from word2vec import TRAIN_DIR
from data_preprocessing import build_dataset
from data_preprocessing import generate_batch

VOCABULARY_SIZE = 50000

print('TEST')

# raw_data = read_data("dataset/DATA/TRAIN", domain_words=1)

words = ['the', 'barometric', 'formula,', 'sometimes', 'called', 'the',
        'exponential', 'atmosphere', 'or', 'isothermal', 'atmosphere,', 'is',
        'a', 'formula', 'used', 'to', 'model', 'how', 'the', 'pressure', '(or',
        'density)', 'of', 'the', 'air', 'changes', 'with', 'altitude.', 'there',
        'are', 'two', 'different', 'equations', 'for', 'computing', 'pressure',
        'at', 'various', 'height', 'regimes', 'below', '86', 'km', '(or',
        '278,400', 'feet).', 'the', 'first', 'equation', 'is', 'used', 'when',
        'the', 'value', 'of', 'standard', 'temperature', 'lapse', 'rate', 'is',
        'not', 'equal', 'to', 'zero;', 'the', 'second', 'equation', 'is',
        'used', 'whenl', 'standard', 'temperature', 'lapse', 'rate', 'equals',
        'zero.']

# data, dictionary, reverse_dictionary = build_dataset(raw_data, VOCABULARY_SIZE)
data, dictionary, reverse_dictionary = build_dataset(words, VOCABULARY_SIZE)
#
#
# print(words[10])
# print(words[15])
# print(len(words))
# for w in words[10:15]:
#     print('hi' + w)

data = [1,2,3,4,5,6,7,8,9]
xs, ys = generate_batch(10, 1, 2, data)
print("DATA:")
print(data)
print("XS:")
print(xs)
print("YS:")
print(ys)
