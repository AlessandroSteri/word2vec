import collections

import numpy as np


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
