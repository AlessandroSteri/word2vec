import os
import time
import numpy as np
import collections
import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree
from sklearn.linear_model import SGDClassifier

from word2vec import TRAIN_DIR
from word2vec import VALID_DIR
from data_preprocessing import data_to_vocab
from data_preprocessing import apply_dictionary
from data_preprocessing import UNK
from utils import LogTime

# from word2vec import data_from_file
from word2vec import get_files_and_domain

import pickle

log_input_dir = "./"
# log_input_dir = "./acc6269"
# log_output_dir = "./"

domain_caching = './log/domain'

def main():
    # exec_id = '152317774459'
    exec_id = '152331714290'
    embedding_size = 128

    # directory = os.path.join(domain_caching, exec_id)
    # caching_directory = os.path.join(domain_caching, 'file_domain', exec_id)
    caching_directory = os.path.join(domain_caching, exec_id)
    # caching_directory = os.path.join(domain_caching, 'file_domain_new')

    num_file_words = 1000000
    num_pec_words = 1000000
    # num_file_to_use = 10 #TODO elimina

    # List of all the files in the training set along with their domains
    with LogTime('Get file paths'):
        training_files = get_files_and_domain(TRAIN_DIR, shuffle=True)

    with LogTime('Restoring env'):
        vocabulary, vectors, dictionary, inv_dictionary, emb_dictionary = recover_execution_environment(exec_id)

    # training_set, labels = build_training_dataset(directory, training_files[:num_file_to_use], vocabulary, emb_dictionary, embedding_size, num_file_words)
    docs_training_set, labels = fetch_docs_with_labels(training_files, caching_directory, caching=True)
    assert len(docs_training_set) == len(labels), "INVALID CALL"

    args = [docs_training_set, labels, vocabulary, emb_dictionary, embedding_size]
    training_set, labels = docs2vec(docs2vec_words_peculiarity_weighted_centroid, *args)


    classifiers = train(training_set, labels)
    validate(classifiers, vocabulary, emb_dictionary, embedding_size, num_file_words)


def train(training_set, labels):
    print("[Training]")
    classifiers = {
        'MLP': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
        # 'SVM': svm.SVC(),
        'NCC': NearestCentroid(),
        'DTR': tree.DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        }
    for name, classifier in classifiers.items():
        with LogTime("Training {}".format(name)):
            print("Training: {}".format(name))
            classifier.fit(training_set, labels)
    return classifiers


def validate(classifiers, vocabulary, emb_dictionary, embedding_size, num_file_words):
    print("[Validation]")
    correct_predictions = {name: 0 for name, classifier in classifiers.items()}
    start = time.time()
    validation_files = None
    # total   = 0
    with LogTime("Get Validation Files ans Domains"):
        validation_files = get_files_and_domain(VALID_DIR, shuffle=True)
    with LogTime("Validation Phase"):
        for i in tqdm.trange(len(validation_files)):
            f, domain = validation_files[i]

            #TODO classifier.score(trainingset, labels)
            centroid_to_validate = file_to_centroid_vec(f, vocabulary, emb_dictionary, embedding_size, num_file_words)

            for name, classifier in classifiers.items():
                prediction = classifier.predict([centroid_to_validate])
                if prediction == domain:
                    correct_predictions[name] = correct_predictions[name] + 1
                if i % 500 == 0:
                    print('[{}] - Predicted: {}, Expected: {}'.format(name, prediction, domain))
                    print('Correct {}: {}, Total: {}, Perc: {}%'.format(name, correct_predictions[name], i+1, int(correct_predictions[name]*100/(i+1))))


def fetch_docs_with_labels(training_files, caching_directory, caching=True):
    docs_training_set = None
    labels = None
    if caching and os.path.exists(caching_directory):
        print('Loading word_test_set and labels from previous same execution.')
        with LogTime('Loading docs_training_set and labels'):
            with open(os.path.join(caching_directory, "word_training_set" + ".pickle"), 'rb') as f:
                docs_training_set = pickle.load(f)
            with open(os.path.join(caching_directory, "labels" + ".pickle"), 'rb') as f:
                labels = pickle.load(f)
    else:
        docs_training_set = []
        labels = []
        for i in tqdm.trange(len(training_files)):
            f, domain = training_files[i]
            file_sentences = data_from_file(f) #, num_file_words)
            docs_training_set.append(file_sentences)
            labels.append(domain)

        if not os.path.exists(caching_directory):
            os.makedirs(caching_directory)
            with LogTime('Caching docs_training_set and labels'):
                with open(os.path.join(caching_directory, "word_training_set" + ".pickle"), 'wb') as f:
                    pickle.dump(docs_training_set, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(caching_directory, "labels" + ".pickle"), 'wb') as f:
                    pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    return docs_training_set, labels

def docs2vec(to_vector, *args):
    print(type(args[0]), type(args[1]))
    print(len(args))
    print(len(args[0]))
    print(len(args[1]))
    training_set, labels = to_vector(*args)
    #TODO call centroid e riaggioustalo che prende doc e rivedi main
    return training_set, labels



def build_training_dataset(directory, training_files, vocabulary, emb_dictionary, embedding_size, num_file_words, num_file_to_use=-1):
    training_set = None
    labels = None
    if os.path.exists(directory):
        print('Loading test_set and labels from previous same execution.')
        with open(os.path.join(directory, "training_set" + ".pickle"), 'rb') as f:
            training_set = pickle.load(f)
        with open(os.path.join(directory, "labels" + ".pickle"), 'rb') as f:
            labels = pickle.load(f)
    elif not os.path.exists(directory): # just in case i add a flag to enable not to cache upon request
        # compute and catch
        training_set = []
        labels = []
        for i in tqdm.trange(len(training_files)):
            f, domain = training_files[i]
            try:
                centroid = file_to_centroid_vec(f, vocabulary, emb_dictionary, embedding_size, num_file_words)
            except ZeroDivisionError as e:
                continue
            training_set.append(centroid)
            labels.append(domain)
            if i == num_file_to_use:
                break
        os.makedirs(directory)
        with open(os.path.join(directory, "training_set" + ".pickle"), 'wb') as f:
            pickle.dump(training_set, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(directory, "labels" + ".pickle"), 'wb') as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    return training_set, labels


def file_to_centroid_vec(file_name, vocabulary, emb_dictionary, embedding_size, num_file_words):
    file_sentences = data_from_file(file_name, num_file_words)
    doc_vocab = data_to_vocab(file_sentences)
    # file_vector = most_peculiar_doc_words(num_pec_words, doc_vocab, vocabulary)
    words, pecurliarities = peculiar_doc_words(doc_vocab, vocabulary)
    # embedded_words = [emb_dictionary[word] for word in words]
    embedded_words = dict()
    try:
        embedded_words = [emb_dictionary[word] for word in words]
    except KeyError as e:
        # raise e
        print("not in emb_dict - should never be printed")
    centroid = mean_vector(embedded_words, pecurliarities, embedding_size)
    return centroid

###
def docs2vec_words_peculiarity_weighted_centroid(docs_training_set, labels, vocabulary, emb_dictionary, embedding_size):
    # print(len(docs_training_set), len(labels))
    assert len(docs_training_set) == len(labels), "INVALID INPUT"
    training_set = []
    new_labels = [] # some docs may be discarded when computing centroid so the label
    for i in tqdm.trange(len(docs_training_set)):
        doc_sentences = docs_training_set[i]
        label = labels[i]
        doc_vocab = data_to_vocab(doc_sentences)
        # file_vector = most_peculiar_doc_words(num_pec_words, doc_vocab, vocabulary)
        words, pecurliarities = peculiar_doc_words(doc_vocab, vocabulary)
        embedded_words = dict()
        try:
            embedded_words = [emb_dictionary[word] for word in words]
        except KeyError as e:
            print("not in emb_dict - should never be printed")
        try:
            centroid = mean_vector(embedded_words, pecurliarities, embedding_size)
            training_set.append(centroid)
            new_labels.append(label)
        except ZeroDivisionError as e:
            continue
        # print(len(training_set), len(new_labels))
        assert len(training_set) == len(new_labels), "INVALID OUTPUT"
    return training_set, new_labels

def data_from_file(file_name, num_file_words=-1):
    limit = num_file_words
    sentences = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            split = line.lower().strip().split()
            if limit > 0 and limit - len(split) < 0:
                split = split[:limit]
            elif limit != -1:
                limit -= len(split)
            if limit >= 0 or limit == -1:
                sentences.append(split)
            # print(sentences)
    return sentences


# occhio a unk!!! devo fare che se non in vocab dataset metto zero in peculiar counter
def most_peculiar_doc_words(num_pec_words, doc_vocab, vocabulary):
    tot_occ_doc = sum(doc_vocab.values())
    tot_occ_dataset = sum(vocabulary.values())
    peculiar_word = collections.Counter()
    for word in doc_vocab:
        occur = doc_vocab[word]
        # assert that doc_vocab is contained in vocabulary
        # print(word, occur)
        if not vocabulary[word]:
            continue
        freq_in_doc = occur / tot_occ_doc
        freq_in_dataset = vocabulary[word] / tot_occ_dataset
        freq_ratio = freq_in_doc / freq_in_dataset
        #TODO was 6, tune to the best working
        precision = 10**9
        peculiarity = int(freq_ratio * precision)
        peculiar_word[word] = peculiarity
        # print(word, peculiarity)
    #TODO check again whats happen with unk not beingin most common
    pecurliar_word_keys = [word for word, occ in peculiar_word.most_common(num_pec_words)]
    return pecurliar_word_keys

def peculiar_doc_words(doc_vocab, vocabulary):
    tot_occ_doc = sum(doc_vocab.values())
    tot_occ_dataset = sum(vocabulary.values())
    peculiar_word = collections.Counter()
    for word in doc_vocab:
        occur = doc_vocab[word]
        # assert that doc_vocab is contained in vocabulary
        # print(word, occur)
        if not vocabulary[word]:
            continue
        freq_in_doc = occur / tot_occ_doc
        freq_in_dataset = vocabulary[word] / tot_occ_dataset
        freq_ratio = freq_in_doc / freq_in_dataset
        precision = 10**6
        peculiarity = int(freq_ratio * precision)
        peculiar_word[word] = peculiarity
        # print(word, peculiarity)
    pecurliar_words = [word for word in peculiar_word]
    pecurliarities = [peculiar_word[word] for word in peculiar_word]
    # pecurliarities = [occ for _, occ in peculiar_word]
    # print(peculiar_word)
    # print(pecurliarities)
    return pecurliar_words, pecurliarities


def recover_execution_environment(exec_id):
    vocabulary = parse_vocab(path_of('vocab', exec_id))
    vector_file = path_of('vectors', exec_id)
    vectors = parse_vectors(vector_file)
    dic_file = path_of('dict', exec_id, ext='.npy')
    dictionary = load_dictionary(dic_file)
    inv_dic_file = path_of('inv_dict', exec_id, ext='.npy')
    inv_dictionary = load_dictionary(inv_dic_file)
    emb_dictionary = embeddings_dictionary(vectors, inv_dictionary)
    return vocabulary, vectors, dictionary, inv_dictionary, emb_dictionary


def parse_vocab(vocab_file):
    vocab = collections.Counter()
    with open(vocab_file) as f:
        for line in f:
            # print(line)
            word, occur = line.split(',')
            vocab[word] = int(occur)
    return vocab


def parse_vectors(vector_file):
    return np.genfromtxt(vector_file, delimiter=',')



def load_dictionary(dic_file):
    dictionary = np.load(dic_file).item()
    return  dictionary


def embeddings_dictionary(vectors, inv_dictionary):
    # assert dictionary len = raw in file vectors
    embeddings_dict = dict()
    # with open(vector_file, 'r') as vectors:
    for index, vector in enumerate(vectors):
            # check if our of one
        embeddings_dict[inv_dictionary[index]] = vector
    return embeddings_dict


def mean_vector(vectors, weights, embedding_size):
    # all vector must have same dimention
    sum_w = sum(weights)
    centroid = [0] * embedding_size
    for v_index, vector in enumerate(vectors):
        for dimention, coordinate in enumerate(vector):
            # __import__('ipdb').set_trace()
            # print('w: {}, c: {}'.format(weights[v_index], coordinate))
            centroid[dimention] += weights[v_index] * coordinate
            # dividi tutto per len vector
    centroid = [ x / sum_w for x in centroid ]
    return centroid

def path_of(entity, exec_id, ext='.csv'):
    path = os.path.join(log_input_dir, 'log/', entity, exec_id + ext)
    return path


if __name__ == '__main__':
    main()

