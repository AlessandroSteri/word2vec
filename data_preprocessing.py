import collections

import numpy as np
import datetime

import nltk
import csv
import string

from time import time

UNK = "<UNK>"
UNK_INDEX = 0

batch_time = 0
batch_index = 0
training_pairs = 0
used_training_pairs = 0
### generate_batch ###
# This function generates the train data and label batch from the dataset.
#
### Parameters ###
# batch_size: the number of train_data,label pairs to produce per batch
# curr_batch: the current batch number.
# window_size: the size of the context
# data: the dataset
### Return values ###
# train_data: train data for current batch
# labels: labels for current batch
def generate_batch(batch_size, curr_batch, window_size, data):
    train_data = []
    labels = []


    ###FILL HERE###
    global batch_time
    global batch_index
    global training_pairs
    global used_training_pairs

    # if curr_batch % 1000 == 0:
    #     print("training_pairs: {}, used_training_pairs: {}".format(training_pairs,
    #           used_training_pairs))
    #     print('Batch time: {}'.format(batch_time))

    start = time()
    # make more elegant
    # full_train_data = []
    # full_labels = []
    # TODO: taken from slides
    # for word_index, pivot_word in enumerate(data[batch_index:]):
    for pivot_word in data[batch_index:]:
        batch_index = batch_index + 1 % len(data)
        window = data[max(batch_index - window_size, 0):]
        window = window[:min(batch_index + window_size, len(data))]
        if len(train_data) >= batch_size:
            break
        for context_word in window:
            if len(train_data) >= batch_size:
                break
            #data[max(batch_index - window_size, 0): min(batch_index +
            #                                    window_size,
            #                                    len(data)) + 1]:
            training_pairs += 1
            if pivot_word != 0 and context_word != 0 and context_word != pivot_word:
                used_training_pairs += 1
                # x, y = skip_gram(pivot_word, context_word)
                train_data.append(pivot_word)
                labels.append(context_word)


    # print("full_train_data: {}".format(len(full_train_data)))
    # print("full_labels: {}".format(len(full_labels)))

    # batch_start_index = batch_size*curr_batch
    # print("batch_start_index {}".format(batch_start_index))
    # batch_end_index   = (batch_size*curr_batch) + batch_size
    # print("batch_end_index {}".format(batch_end_index))
    # train_data        = full_train_data[batch_start_index:batch_end_index]
    # labels            = full_labels[batch_start_index:batch_end_index]
    # print("train_data: {}".format(train_data))
    # print("labels: {}".format(labels))


    # data_len = len(data)
    # # curr_batch ranges from 0 to num_step-1
    # start = (curr_batch * batch_size) % data_len  # start is included in this batch
    #
    # # catch for current batch from data (as circular buffer)
    # batch = []
    # for i in range(batch_size):
    #     # i = 0...batch_size-1
    #     index = (start + i) % data_len
    #     batch.append(data[index])
    #
    # train_data = []
    # labels = []
    # # TODO: taken from slides
    # for word_index, word in enumerate(batch):
    #     for nb_word in batch[max(word_index - window_size, 0): min(word_index + window_size, len(batch)) + 1]:
    #         if nb_word != word:
    #             x, y = skip_gram(word, nb_word)
    #             train_data.append(x)
    #             labels.append(y)
    #
    # print("Train data len: {}".format(len(train_data)))
    # train_data = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 ]
    # labels = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 ]

    train_data = np.asarray(train_data)
    labels = np.asarray(labels).reshape(batch_size,1)

    # print("Train data shape: {}".format(train_data.shape))

    stop = time()
    dur = stop - start
    batch_time += dur
    # print('time spent in batch: {}'.format(batch_time))
    return train_data, labels

### build_dataset ###
# This function is responsible of generating the dataset and dictionaries.
# While constructing the dictionary take into account the unseen words by
# retaining the rare (less frequent) words of the dataset from the dictionary
# and assigning to them a special token in the dictionary: UNK. This
# will train the model to handle the unseen words.
### Parameters ###
# words: a list of words
# vocab_size:  the size of vocabulary
#
### Return values ###
# data: list of codes (integers from 0 to vocabulary_size-1).
#       This is the original text but words are replaced by their codes
# dictionary: map of words(strings) to their codes(integers)
# reverse_dictionary: maps codes(integers) to words(strings)
def build_dataset(words, vocab_size):
    dictionary = dict()
    reversed_dictionary = dict()
    data = None

    ###FILL HERE###

    stopwords_file = './stopwords2.txt'
    # stopwords_file = './stopwords_full.txt'
    #TODO: taken from course slides:
    stopwords = get_stopwords(stopwords_file)
    vocab = collections.Counter()

    # tokenizer = nltk.tokenizer.casual.casual
    for w in words:
        if w == '' or w in stopwords:
            # NB data will also skip those
            continue
        if '\\' in w:
            # remove wiki markdown
            # print('[build_dataset] Word with \ : {}'.format(w))
            continue
        if any(c.isdigit() for c in w):
            # print('[build_dataset] Word with digit: {}'.format(w))
            continue
        if len(w) <= 2:
            # remove most of stop words, remove also up which is ineresting
            # print('[build_dataset] Word with less than 2: {}'.format(w))
            continue

        char_to_take_of = set(string.punctuation)
        # allow word like new york
        # TODO: remove -.
        # char_to_take_of.pop('-')
        # allow u.s.a.
        # char_to_take_of.pop('.')
        w_no_char = "".join(char for char in w if char not in char_to_take_of)
        if '..' not in w_no_char:
            w_no_char =  w_no_char.rstrip('-.').lstrip('-.')
        else:
            # preserving acronyms
            w_no_char =  w_no_char.rstrip('-').lstrip('-')
            w_no_char = w_no_char.replace('..', '.')


        vocab[w_no_char] += 1
    # print("Distinct words: ", len(vocab))
    #TODO: taken from course slides^
    # TODO: taken from stackoverflow
    file_name = 'out/' + str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.') + '_vocabulary.csv'
    # file_name = 'vocabulary.csv'
    # file_name = time.time.strftime('%Y%m%d-%H%M%S') + '- vocabulary' + '.csv'
    with open(file_name,'w') as csvfile:
        fieldnames=['word','occur']
        writer=csv.writer(csvfile)
        writer.writerow(fieldnames)
        for key, value in enumerate(vocab.most_common(vocab_size-1)):

            writer.writerow([value[0], value[1]])
    # Is a bias the fact that we follow the most common word order?
    for index, word_occurrency in enumerate(vocab.most_common(vocab_size-1)):
        # vocab_size -1 to leave one spot for UNK

        common_word, _ = word_occurrency
        # print("Index: {}, word: {}".format(index, common_word))

        dictionary[common_word] = index + 1
        reversed_dictionary[index + 1] = common_word

    # Handling less frequent words as UNK
    dictionary[UNK] = UNK_INDEX #vocab_size-1
    reversed_dictionary[UNK_INDEX] = UNK
    #TODO: double check if UNK is out of one


    data = apply_dictionary(words, dictionary, UNK)
    # uncomment to debug with words instead of indexes
    # data = words

    return data, dictionary, reversed_dictionary

###
# Save embedding vectors in a suitable representation for the domain identification task
###
def save_vectors(vectors):

    ###FILL HERE###
    file_name = 'out/' + str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.') + '_vectors.csv'
    # file_name = 'vector.csv'
    # file_name = time.strftime('%Y%m%d-%H%M%S') + ' - embedding' + '.csv'
    print(vectors)
    print(vectors.shape)
    print(vectors[1].shape)
    np.savetxt(file_name, vectors, delimiter=',')

    pass


# Reads through the analogy question file.
#    Returns:
#      questions: a [n, 4] numpy array containing the analogy question's
#                 word ids.
#      questions_skipped: questions skipped due to unknown words.
#
def read_analogies(file, dictionary):
    questions = []
    questions_skipped = 0
    with open(file, "r") as analogy_f:
        for line in analogy_f:
            if line.startswith(":"):  # Skip comments.
                continue
            words = line.strip().lower().split(" ")
            ids = [dictionary.get(str(w.strip()), 0) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
    print("Eval analogy file: ", file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)

### LOADING STOPWORDS ### #TODO: taken from course slides
def get_stopwords(file):
    return set([w.rstrip('\r\n') for w in open(file)])


def apply_dictionary(x_list, dictionary, unk_key):
    y_list = []
    for x in x_list:
        y = None
        try:
            y = dictionary[x]
            # y_list.append(y)
        except KeyError:
            y = dictionary[unk_key]

        y_list.append(y)
    return y_list

### MY HELPER FUNCTIONS ###
def skip_gram(word, nb_word):
    return word, nb_word

def cbow(word, nb_word):
    return nb_word, word


# def negative_sample(num_negative, word, dictionary):
    # pass

