import collections

import numpy as np

UNK = "UNK"

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
    train_data = None
    labels = None

    ###FILL HERE###

    data_len = len(data)
    # curr_batch ranges from 0 to num_step-1
    start = (curr_batch * batch_size) % data_len  # start is included in this batch

    # catch for current batch from data (as circular buffer)
    batch = []
    for i in range(batch_size):
        # i = 0...batch_size-1
        index = (start + i) % data_len
        batch.append(data[index])

    train_data = []
    labels = []
    # TODO: taken from slides
    for word_index, word in enumerate(batch):
        for nb_word in batch[max(word_index - window_size, 0): min(word_index + window_size, len(batch)) + 1]:
            if nb_word != word:
                x, y = skip_gram(word, nb_word)
                train_data.append(x)
                labels.append(y)

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

    #TODO: taken from course slides:
    stopwords = get_stopwords('./stopwords.txt')
    vocab = collections.Counter()

    for w in words:
        if w == '' or w in stopwords:
            # NB data will also skip those
            continue
        vocab[w] += 1
    print("Distinct words: ", len(vocab))
    #TODO: taken from course slides^

    # Is a bias the fact that we follow the most common word order?
    for index, word_occurrency in enumerate(vocab.most_common(vocab_size-1)):
        # vocab_size -1 to leave one spot for UNK

        common_word, _ = word_occurrency
        print("Index: {}, word: {}".format(index, common_word))

        dictionary[common_word] = index
        reversed_dictionary[index] = common_word

    # Handling less frequent words as UNK
    dictionary[UNK] = vocab_size
    reversed_dictionary[vocab_size] = UNK
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
        except KeyError:
            y = dictionary[unk_key]
        y_list.append(y)
    return y_list

### MY HELPER FUNCTIONS ###
def skip_gram(word, nb_word):
    return word, nb_word

def cbow(word, nb_word):
    return nb_word, word

