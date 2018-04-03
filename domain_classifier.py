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


def main():
    exec_id = '152269643520'
    embedding_size = 200

    num_file_words = 1000000
    num_pec_words = 1000000

    # list of all the files in the training set
    training_files = get_files_and_domain(TRAIN_DIR, shuffle=True)

    vocabulary, vectors, dictionary, inv_dictionary, emb_dictionary = recover_execution_environment(exec_id)

    trainig_set, labels = build_training_dataset(training_files, vocabulary, emb_dictionary, embedding_size, num_file_words, num_file_to_use=1000)
    # num_files = 10000


    print("Training")
    clf_mlp = training_mlp(trainig_set, labels)
    clf_svm = svm.SVC()
    clf_svm.fit(trainig_set, labels)
    clf_ncc = NearestCentroid()
    clf_ncc.fit(trainig_set, labels)
    clf_dtr = tree.DecisionTreeClassifier()
    clf_dtr.fit(trainig_set, labels)
    clf_sgd = SGDClassifier(loss="hinge", penalty="l2")
    clf_sgd.fit(trainig_set, labels)
    clf_ncccs = NearestCentroid(metric='l2')
    clf_ncccs.fit(trainig_set, labels)
    print("Validation")
    validation_files = get_files_and_domain(VALID_DIR, shuffle=True)
    correct_mlp = 0
    correct_svm = 0
    correct_ncc = 0
    correct_dtr = 0
    correct_sgd = 0
    correct_ncccs = 0
    correct_my = 0
    total   = 0
    for i in tqdm.trange(len(validation_files)):
        # if i == num_files:
            # break
        f, domain = validation_files[i]
        centroid_to_validate = file_to_centroid_vec(f, vocabulary, emb_dictionary, embedding_size, num_file_words)
        prediction_mlp = clf_mlp.predict([centroid_to_validate])
        prediction_svm = clf_svm.predict([centroid_to_validate])
        prediction_ncc = clf_ncc.predict([centroid_to_validate])
        prediction_dtr = clf_dtr.predict([centroid_to_validate])
        prediction_sgd = clf_sgd.predict([centroid_to_validate])
        prediction_ncccs = clf_ncccs.predict([centroid_to_validate])
        my_prediction = prediction_ncc
        if prediction_sgd == prediction_dtr or prediction_sgd == prediction_mlp or prediction_sgd == prediction_svm:
            my_prediction = prediction_sgd
        total += 1
        if prediction_mlp == domain:
            correct_mlp += 1
        if prediction_svm == domain:
            correct_svm += 1
        if prediction_ncc == domain:
            correct_ncc += 1
        if prediction_dtr == domain:
            correct_dtr += 1
        if prediction_sgd == domain:
            correct_sgd += 1
        if prediction_ncccs == domain:
            correct_ncccs += 1
        if my_prediction == domain:
            correct_my += 1
        print('[MLP] - Predicted: {}, Expected: {}'.format(prediction_mlp, domain))
        print('Correct_mlp: {}, Total: {}, Perc: {}%'.format(correct_mlp, total, int(correct_mlp*100/total)))
        print('[SVM] - Predicted: {}, Expected: {}'.format(prediction_svm, domain))
        print('Correct_svm: {}, Total: {}, Perc: {}%'.format(correct_svm, total, int(correct_svm*100/total)))
        print('[NCC] - Predicted: {}, Expected: {}'.format(prediction_ncc, domain))
        print('Correct_ncc: {}, Total: {}, Perc: {}%'.format(correct_ncc, total, int(correct_ncc*100/total)))
        print('[DTR] - Predicted: {}, Expected: {}'.format(prediction_dtr, domain))
        print('Correct_dtr: {}, Total: {}, Perc: {}%'.format(correct_dtr, total, int(correct_dtr*100/total)))
        print('[SGD] - Predicted: {}, Expected: {}'.format(prediction_sgd, domain))
        print('Correct_sgd: {}, Total: {}, Perc: {}%'.format(correct_sgd, total, int(correct_sgd*100/total)))
        print('[NCCCS] - Predicted: {}, Expected: {}'.format(prediction_ncccs, domain))
        print('Correct_ncccs: {}, Total: {}, Perc: {}%'.format(correct_ncccs, total, int(correct_ncccs*100/total)))
        print('[MY] - Predicted: {}, Expected: {}'.format(my_prediction, domain))
        print('Correct_my: {}, Total: {}, Perc: {}%'.format(correct_my, total, int(correct_my*100/total)))


def build_training_dataset(training_files, vocabulary, emb_dictionary, embedding_size, num_file_words, num_file_to_use=-1):
    trainig_set = []
    labels = []
    for i in tqdm.trange(len(training_files)):
        f, domain = training_files[i]
        try:
            centroid = file_to_centroid_vec(f, vocabulary, emb_dictionary, embedding_size, num_file_words)
        except ZeroDivisionError as e:
            continue
        trainig_set.append(centroid)
        labels.append(domain)
        if i == num_file_to_use:
            break
    return trainig_set, labels


def file_to_centroid_vec(file_name, vocabulary, emb_dictionary,embedding_size, num_file_words):
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


def data_from_file(file_name, num_file_words=-1):
    limit = num_file_words
    sentences = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            split = line.lower().strip().split()
            if limit > 0 and limit - len(split) < 0:
                split = split[:limit]
            else:
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
        precision = 10**6
        peculiarity = int(freq_ratio * precision)
        peculiar_word[word] = peculiarity
        # print(word, peculiarity)
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

def get_files_and_domain(directory, shuffle=False):
    files = []
    # data = []
    for domain in os.listdir(directory):
    #for dirpath, dnames, fnames in os.walk(directory):
        # limit = domain_words
        # Compatibility with macOS
        if domain == ".DS_Store":
            continue
        for f in os.listdir(os.path.join(directory, domain)):
            file_path = os.path.join(directory, domain, f)
            if f.endswith(".txt"):
                files.append((file_path, domain))
            # print(file_path)
            # if f.endswith(".txt"):
            #     with open(os.path.join(directory, domain, f)) as file:
            #         # sentences = []
            #         for line in file.readlines():
            #             split = line.lower().strip().split()
            #             if limit > 0 and limit - len(split) < 0:
            #                 split = split[:limit]
            #             else:
            #                 limit -= len(split)
            #             if limit >= 0 or limit == -1:
            #                 data.append(split)

    files = np.asarray(files, dtype=str)
    # print(len(files))
    if shuffle:
        start = time.time()
        np.random.shuffle(files)
        stop = time.time()
        # print('Shuffle: ', stop-start)
    return files


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
    path = os.path.join('./log/', entity, exec_id + ext)
    return path

def training_mlp(trainig_set, labels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(trainig_set, labels)
    return clf

if __name__ == '__main__':
    main()

