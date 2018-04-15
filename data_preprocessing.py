import os
import collections

import numpy as np

import nltk
import csv
import string
import re
from time import time
import ipdb
import pickle


UNK = "<UNK>"
UNK_INDEX = 0
STOPWORDS_FILE = './stopwords2.txt'

### generate_batch ###
# This function generates the train data and label batch from the dataset.
def generate_batch(batch_size, curr_sentence, curr_word, curr_context_word, window_size, data):
    train_data = []
    labels = []

    start = time()

    processed_sentences = curr_sentence
    processed_words = curr_word
    processed_context_words = curr_context_word
    while len(train_data) < batch_size:
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
            # Other words to process in current sentence
            window = None
            window = sentence[max(processed_words - window_size,0):min(processed_words + window_size,len(sentence)) + 1]
            # print('processed_words: {}, window_size:{}, len window:{}'.format(processed_words, window_size, len(window)))
            # print('Window: {}'.format(window))
            if processed_context_words == len(window):
                processed_words += 1
                processed_context_words = 0
                continue
            else:
                context_word = window[processed_context_words]
                processed_context_words +=1
                # training_pairs += 1
                pivot_word = sentence[processed_words]
                # print('PW: {}, CW: {}'.format(pivot_word, context_word))
                if pivot_word != UNK_INDEX and context_word != UNK_INDEX and context_word != pivot_word:
                    train_data.append(pivot_word)
                    labels.append(context_word)
                    # print('Train data:\n{}'.format(train_data))
                    # print('Labels:\n{}'.format(labels))

    train_data = np.asarray(train_data)
    labels = np.asarray(labels).reshape(batch_size,1)

    stop = time()
    dur = stop - start
    return train_data, labels, processed_sentences,processed_words, processed_context_words



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
def build_dataset(sentences, vocab_size, execution_id):
    dictionary = dict()
    reversed_dictionary = dict()
    data = None

    ###FILL HERE###
    start = time()
    vocab = data_to_vocab(sentences)
    print("Data to vocab: {}".format(time() - start))

    # save vocab to csv file
    start = time()
    vocab_to_csv(vocab, vocab_size, execution_id)
    print("Vocab to csv: {}".format(time() - start))

    # Build dictionary
    start = time()
    for index, word_ in enumerate(vocab.most_common(vocab_size-1)): # vocab_size -1 to leave one spot for UNK
        word, _ = word_
        # print("Index: {}, word: {}".format(index, word))

        dictionary[word] = index + 1
        reversed_dictionary[index + 1] = word

    # Handling less frequent words as UNK
    dictionary[UNK] = UNK_INDEX
    reversed_dictionary[UNK_INDEX] = UNK
    print("Build dictionary: {}".format(time() - start))

    data = []
    start = time()
    for sentence in sentences:
        # sentence_2_int = sentence # text in clear for debugging
        sentence_2_int = apply_dictionary(sentence, dictionary, UNK)
        data.append(sentence_2_int)

    print("Sentences: {}".format(time() - start))

    return data, dictionary, reversed_dictionary, vocab

###
# Save embedding vectors in a suitable representation for the domain identification task
###
def save_vectors(vectors, execution_id):

    ###FILL HERE###
    start = time()
    file_name = os.path.join('./log/vectors', str(execution_id) + '.csv')
    np.savetxt(file_name, vectors, delimiter=',')
    stop = time()
    start_p = time()
    with open(os.path.join('./log/vectors', str(execution_id) + '.pickle'), 'wb') as vector_file:
        pickle.dump(vectors, vector_file, protocol=pickle.HIGHEST_PROTOCOL)
    stop_p = time()
    print("Save vector time, csv: {} sec vs pickle: {} sec.".format(stop - start, stop_p - start_p))

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
            # ids = [dictionary.get(str(w.strip()), 0) for w in words]
            ids = [dictionary.get(str(w.strip())) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
    print("Eval analogy file: ", file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)

### LOADING STOPWORDS ###
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
def data_to_vocab(sentences):
    stopwords = get_stopwords(STOPWORDS_FILE)
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
            w_no_char = "".join(char for char in w if char not in char_to_take_of)
            if '..' not in w_no_char:
                w_no_char =  w_no_char.rstrip('-.').lstrip('-.')
            else:
                # preserving acronyms
                w_no_char =  w_no_char.rstrip('-').lstrip('-')
                w_no_char = w_no_char.replace('..', '.')


            if len(w_no_char) <= 2 or w_no_char in stopwords:
                continue
            vocab[w_no_char] += 1
    return vocab


def vocab_to_csv(vocab, vocab_size, execution_id):
    file_name = os.path.join('./log/vocab', str(execution_id) + '.csv')
    with open(file_name,'w') as f:
        writer=csv.writer(f)
        for key, value in enumerate(vocab.most_common(vocab_size-1)):
            writer.writerow([value[0], value[1]])


def get_training_set_coverage(batch_size, num_steps, training_set_cardinality):
    coverage =  (batch_size * num_steps * 100) / training_set_cardinality
    epoch =  (batch_size * num_steps) / training_set_cardinality
    training_pairs = batch_size * num_steps
    return training_pairs, epoch, coverage, training_set_cardinality

def compute_training_set_cardinality(window_size, data):
    num_training_pairs = 0
    start = time()
    for sentence in data:
            # Other words to process in current sentence
        for pivot_word in sentence:
            window = sentence[max(pivot_word - window_size, 0):min(pivot_word + window_size, len(sentence)) + 1]
            for context_word in window:
                if pivot_word != UNK_INDEX and context_word != UNK_INDEX and context_word != pivot_word:
                    num_training_pairs += 1

    stop = time()
    dur = stop - start
    print('Time spent computing training set cardinality: {}'.format(dur))
    return num_training_pairs
