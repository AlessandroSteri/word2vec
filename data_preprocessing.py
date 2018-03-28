import collections

import numpy as np
import datetime

import nltk
import csv
import string
import re
from time import time
import ipdb
UNK = "<UNK>"
UNK_INDEX = 0

batch_time = 0
batch_index = 0
training_pairs = 0
used_training_pairs = 0
### generate_batch ###
# This function generates the train data and label batch from the dataset.
#
### Parameters ### TODO: update con parametri nuovi
# batch_size: the number of train_data,label pairs to produce per batch
# curr_batch: the current batch number.
# window_size: the size of the context
# data: the dataset
### Return values ###
# train_data: train data for current batch
# labels: labels for current batch
# def generate_batch(batch_size, curr_batch, window_size, data):
def generate_batch(batch_size, curr_sentence, curr_word, curr_context_word, window_size, data):
    print('Generate')
    train_data = []
    labels = []


    ###FILL HERE###
    global batch_time
    # global batch_index
    global training_pairs
    global used_training_pairs

    # if curr_batch % 1000 == 0:
    #     print("training_pairs: {}, used_training_pairs: {}".format(training_pairs,
    #           used_training_pairs))
    #     print('Batch time: {}'.format(batch_time))

    start = time()
    # TODO: taken from slides
    # for word_index, pivot_word in enumerate(data[batch_index:]):

    processed_sentences = curr_sentence
    processed_words = curr_word
    processed_context_words = curr_context_word
    while len(train_data) < batch_size:
        # if len(train_data) == 5:
        # ipdb.set_trace()
        # print('Pivot:\n{}'.format(processed_words))
        sentence_index = processed_sentences % len(data)
        sentence = data[sentence_index]
        if processed_words == len(sentence):
            # print('end sentence')
            processed_sentences += 1
            processed_words = 0
            processed_context_words = 0
            continue
        else:
            # print('inside sentence: {}'.format(processed_sentences))
            # Other words to process in current sentence
            window = None
            window = sentence[max(processed_words - window_size,0):min(processed_words + window_size,len(sentence))]# + 1]
            print('processed_words: {}, window_size:{}, len window:{}'.format(processed_words, window_size, len(window)))
            # print('Window: {}'.format(window))
            if processed_context_words == len(window):
                # print('end window')
                processed_words += 1
                processed_context_words = 0
                continue
            else:
                context_word = window[processed_context_words]
                processed_context_words +=1
                training_pairs += 1
                pivot_word = sentence[processed_words]
                # print('PW: {}, CW: {}'.format(pivot_word, context_word))
                if pivot_word != UNK_INDEX and context_word != UNK_INDEX and context_word != pivot_word:
                    used_training_pairs += 1
                    train_data.append(pivot_word)
                    labels.append(context_word)
                    # print('Train data:\n{}'.format(train_data))
                    # print('Labels:\n{}'.format(labels))

    train_data = np.asarray(train_data)
    labels = np.asarray(labels).reshape(batch_size,1)

    # print("Train data shape: {}".format(train_data.shape))

    stop = time()
    dur = stop - start
    batch_time += dur
    print('time spent in batch: {}'.format(batch_time))
    return train_data, labels

    # for sentence in data[curr_sentence:]: # if last sentence?
    #     # if len(sentence) == 1:
    #         # continue
    #     for pivot_word in sentence[curr_word:]:
    #         # if len(train_data) >= batch_size:
    #             # break
    #         # batch_index = batch_index + 1 % len(data)
    #         window = sentence[max(w_idx - window_size, 0):]
    #         window = window[:min(w_idx + window_size, len(data))]
    #         for context_word in window[curr_context_word:]:
    #             if len(train_data) >= batch_size:
    #                 # incrementi?
    #                 break
    #             training_pairs += 1
    #             if pivot_word != UNK_INDEX and context_word != UNK_INDEX and context_word != pivot_word:
    #                 used_training_pairs += 1
    #                 train_data.append(pivot_word)
    #                 labels.append(context_word)
    #

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
def build_dataset(sentences, vocab_size):
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
    for sentence in sentences:
        for w in sentence:
            if w == '': # or w in stopwords:
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
            if w == '' or w in stopwords:
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


            if len(w_no_char) <= 2 or w_no_char in stopwords:
                # remove most of stop words, remove also up which is ineresting
                # print('[build_dataset] Word with less than 2: {}'.format(w))
                continue
            vocab[w_no_char] += 1
    # print("Distinct words: ", len(vocab))

    #TODO: taken from course slides^
    # TODO: taken from stackoverflow
    file_name = 'out/' + str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.') + '_vocabulary.csv'
    # file_name = 'vocabulary.csv'
    with open(file_name,'w') as csvfile:
        fieldnames=['word','occur']
        writer=csv.writer(csvfile)
        writer.writerow(fieldnames)
        for key, value in enumerate(vocab.most_common(vocab_size-1)):
            writer.writerow([value[0], value[1]])

    # Build dictionary
    for index, word_occurrency in enumerate(vocab.most_common(vocab_size-1)):
        # vocab_size -1 to leave one spot for UNK

        common_word, _ = word_occurrency
        # print("Index: {}, word: {}".format(index, common_word))

        dictionary[common_word] = index + 1
        reversed_dictionary[index + 1] = common_word

    # Handling less frequent words as UNK
    dictionary[UNK] = UNK_INDEX #vocab_size-1
    reversed_dictionary[UNK_INDEX] = UNK


    data = []
    for sentence in sentences:
        sentence_2_int = sentence #apply_dictionary(sentence, dictionary, UNK)
        data.append(sentence_2_int)

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


### apply_dictionary ###
# This function maps elements of a list according to a given dictionary.
#
### Parameters ###
# x_list: list of elements to map
# dictionary: defines the map x -> y to apply
# unk_key: default y for xs not in the domain of the map
### Return values ###
# y_list: list of x in x_list mapped accordingly to dictionary
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

# def is_acronym():
    # acr = r'(?:[a-z]\.)+'
