import os
import sys
from utils import LogTime
from word2vec import get_files_and_domain, TRAIN_DIR, VALID_DIR, log
from data_preprocessing import data_to_vocab
import pickle
from collections import Counter
import numpy as np
import tqdm
from statistics import mean
import math
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import copy
from time import time
import matplotlib.pyplot as plt

TEST_DIR       = "dataset/DATA/TEST"
log_input_dir  = "./"
domain_caching = './log/domain'
out_dir        = 'test_answ'


# TODO: post hw, riattiva autofolding in vimrc, unplug ycm and pymode (lento)

def main(execution_id, num_training_file_s, num_validation_file_s, num_test_file_s):
    # execution_id: identifier for word2vec.train ececution
    # num_training_file_s: number of training files to consider in training
    # num_validation_file_file_s: number of files to consider in validation
    # num_test_file_file_s: number of test files to compute answer for

    # log
    d_id = int(time())
    num_training_file = int(num_training_file_s) #if -1 leaves out last file, trascurable
    num_validation_file = int(num_validation_file_s) #if -1 leaves out last file, trascurable
    num_test_file = int(num_test_file_s) #if -1 uses all test files
    to_log = 'Domain Execution: {}, num_training_file: {}, num_validation_file: {}, num_test_file: {}\n'.format(d_id, num_training_file, num_validation_file, num_test_file)
    log(os.path.join(out_dir, 'log.txt'), to_log)

    # where to save test answers
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Execution id to retrieve environment via id and to cache data in custom folder
    exec_id = str(execution_id)

    # were to dump objects which depends both on embeddings and num_files
    caching_directory = os.path.join(domain_caching, exec_id, str(num_training_file))
    if not os.path.exists(caching_directory):
        os.makedirs(caching_directory)

    # keep track of confusion matrixes
    conf_matrix_dir = 'test_answ/conf_matrix/{}'.format(d_id)
    if not os.path.exists(conf_matrix_dir):
        os.makedirs(conf_matrix_dir)

    # were to dump objects which depends only on num_files
    caching_num_tr_files_directory = os.path.join(domain_caching, str(num_training_file))
    if not os.path.exists(caching_num_tr_files_directory):
        os.makedirs(caching_num_tr_files_directory)

    # LogTime: just an utility I defined to print time needed to complete indent block for optimization purpose.
    with LogTime('Get file paths'):
        # Shuffled list of all the files in the training set along with their domains
        training_files = get_files_and_domain(TRAIN_DIR, shuffle=True)

    with LogTime('Restoring env'):
        # recover all the relevant objects from the choosen execution of word2vec.train()
        vocabulary, vectors, dictionary, inv_dictionary, emb_dictionary, embedding_size = recover_execution_environment(exec_id)

    # with LogTime('Domain stats'):
        # num_docs, domain_doc_number = get_domain_doc_stats()

    # fetches the training set in clear (docs = sentences of words) and labels
    # docs_training_set, labels = fetch_docs_with_labels(training_files, caching_directory, caching=False)
    docs_training_set, labels = fetch_docs_with_labels(training_files[:num_training_file], caching_num_tr_files_directory, caching=False)

    # compute vocabulary of each domain, caching indipendent from execution used
    vocab_caching_dir = os.path.join(domain_caching, str(num_training_file))
    ds_vocabs = domains_vocabularies(docs_training_set, labels, vocabulary, vocab_caching_dir)

    # form domain vocabulary compute centroid, max and min pointwise vector for each domain: dict[domain] = (centr, min, max, avg)
    domains_centroid_max_min_vectors_dict = domains_centroid_max_min_vectors(ds_vocabs, vocabulary, embedding_size, emb_dictionary, caching_directory)

    ### from word-made training set to embedding_vector-made training set ###
    args = [docs_training_set, labels, vocabulary, emb_dictionary, embedding_size, dictionary, domains_centroid_max_min_vectors_dict]
    training_set, labels = docs2vec(docs2vec_words_peculiarity_weighted_centroid, *args)

    ### traing various model ###
    classifiers = train(training_set, labels)


    ### validation ###
    max_accuracy, max_acc_name = skl_validate(classifiers, vocabulary, emb_dictionary, embedding_size, dictionary, domains_centroid_max_min_vectors_dict, num_validation_file, conf_matrix_dir)

    if max_accuracy > 0.7:
        with LogTime('Caching classifiers'):
            with open(os.path.join(caching_directory, "classifier_{}_acc_{}_ntf_{}_nvf_{}".format(max_acc_name, max_accuracy, num_training_file, num_validation_file) + ".pickle"), 'wb') as f:
                pickle.dump(classifiers[max_acc_name], f, protocol=pickle.HIGHEST_PROTOCOL)


    ### test ###
    test_classifiers(max_acc_name, classifiers[max_acc_name], vocabulary, emb_dictionary, embedding_size, dictionary, domains_centroid_max_min_vectors_dict, max_accuracy, num_test_file, d_id)

    to_log = '##################################### DONE ######################################\n'
    log(os.path.join(out_dir, 'log.txt'), to_log)

# returns trained classifier in a dictionary name->trained_classifier
def train(training_set, labels):
    print("[Training]")
    # a buch of classifier to train
    classifiers = {
        # 'MLP': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
        'MLP2': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 40), random_state=1),
        'KNN':KNeighborsClassifier(n_jobs=-1),
        'RNDF':RandomForestClassifier(n_estimators=10, n_jobs =-1),
        'RNDF100':RandomForestClassifier(n_estimators=100, n_jobs =-1),
        'RNDF300':RandomForestClassifier(n_estimators=300, n_jobs =-1),
        'RNDF500':RandomForestClassifier(n_estimators=500, n_jobs =-1),
        'LSVM': LinearSVC(),
        # 'SVM': svm.SVC(),
        'NCC': NearestCentroid(),
        # 'DTR': tree.DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2", n_jobs=-1),
        }
    # # use all classifiers as estimators
    # estimators = [(n, c) for n, c in classifiers.items()]
    # clf = VotingClassifier(estimators, n_jobs=-1)
    # classifiers['ENS'] = clf
    for name, classifier in classifiers.items():
        # log training time for each classifier
        with LogTime("Training {}".format(name)):
            print("Training: {}".format(name))
            classifier.fit(training_set, labels)
    return classifiers


# validate classifier on dev set to tune hyperparameters, select best classifier and compute f1, conf matrix and accuracy
def skl_validate(classifiers, vocabulary, emb_dictionary, embedding_size, dictionary, domains_centroid_max_min_vectors_dict, num_validation_file, conf_matrix_dir):
    print("[Validation]")
    # keep track of best classifier
    max_accuracy = -1
    max_acc_name = None
    validation_files = get_files_and_domain(VALID_DIR, shuffle=True)
    # docs_test_set, labels = fetch_docs_with_labels(validation_files, 'I_DO_NOT_EXIST', caching=False)
    docs_test_set, labels = fetch_docs_with_labels(validation_files[:num_validation_file], 'I_DO_NOT_EXIST', caching=False)
    assert len(docs_test_set) == len(labels), "INVALID CALL"
    args = [docs_test_set, labels, vocabulary, emb_dictionary, embedding_size, dictionary, domains_centroid_max_min_vectors_dict]
    test_set, labels = docs2vec(docs2vec_words_peculiarity_weighted_centroid, *args)
    assert len(test_set) == len(labels), "INVALID CALL"
    domains_counter = Counter(labels)
    class_names = list(domains_counter)
    classifiers_stats = dict()
    for name, classifier in classifiers.items():
        with LogTime(name):
            domains = list(set(labels))
            predictions = classifier.predict(test_set)
            err = sum(1 for pr, lb in zip(predictions, labels) if pr != lb)
            errs_perc = (err / len(predictions))*100
            score = f1_score(labels, predictions, average='macro')
            score_w = f1_score(labels, predictions, average='weighted')
            score_d = f1_score(labels, predictions, average=None, labels=domains)
            recall = recall_score(labels, predictions, average='weighted')
            recall_d = recall_score(labels, predictions, average=None, labels=domains)
            precision_d = precision_score(labels, predictions, average=None, labels=domains)
            precision_w = precision_score(labels, predictions, average='weighted')
            precision_mi = precision_score(labels, predictions, average='micro')
            precision_ma = precision_score(labels, predictions, average='macro')
            accu_sc = accuracy_score(labels, predictions, normalize=False)
            accu_sc_norm = accuracy_score(labels, predictions, normalize=True)

            classifiers_stats[name] = (name, score, score_w, recall, errs_perc, precision_w, precision_mi, precision_ma, accu_sc, accu_sc_norm)
            stats = '\t[{}] f1: {}, f1_w: {}, recall: {}, err%: {}, prec_w: {}, prec_mic: {}, prec_mac: {}, acc_sc: {}, acc_sc_norm: {}\n'.format(name, score, score_w, recall, errs_perc, precision_w, precision_mi, precision_ma, accu_sc, accu_sc_norm)
            log(os.path.join(out_dir, 'log.txt'), stats)
            for d, f1, rc, pr in zip(domains, score_d, recall_d, precision_d):
                log(os.path.join(out_dir, 'log.txt'), '\t\t[{}] F1: {}, Recall: {}, Precision: {}\n'.format(d, f1, rc, pr))


            # Compute confusion matrix
            cnf_matrix = confusion_matrix(labels, predictions)
            np.set_printoptions(threshold=np.nan)

            cm_file = os.path.join(conf_matrix_dir, 'conf_matrix_{}.csv'.format(name))
            with open(cm_file, 'w') as f:
                # np.savetxt(f, cnf_matrix.astype(int), delimiter=',')
                f.write(np.array2string(cnf_matrix, separator=','))
            # with open('conf_matrix_{}_norm.csv'.format(name), 'w') as f:
                # np.savetxt(f, vectors, delimiter=',')
                # f.write(np.array2string(cnf_matrix, separator=','))
            acc = classifier.score(test_set, labels)
            if acc > max_accuracy:
                max_accuracy = acc
                max_acc_name = name
            print('[{}] Score (accuracy): {}, F1_Score_w: {}, F1_Score: {}'.format(name, acc, score_w, score))
            print('[{}] F1 Score: \n{}'.format(name, score))
            print('[{}] f1: {}, f1_w: {}, recall: {}, err%: {}, prec_w: {}, prec_mic: {}, prec_mac: {}, accu_sc: {}, accu_sc_norm: {}\n'.format(name, score, score_w, recall, errs_perc, precision_w, precision_mi, precision_ma, accu_sc, accu_sc_norm))
            # print('[{}] Confusion Matrix: \n{}'.format(name, cnf_matrix))
    stats = '\tBEST CLASSIFIER [{}] f1: {}, f1_w: {}, recall: {}, err%: {}%, prec_w: {}, prec_mic: {}, prec_mac: {}, acc_sc: {}, acc_sc_norm: {}\n'.format(*classifiers_stats[max_acc_name])
    log(os.path.join(out_dir, 'log.txt'), stats)
    return max_accuracy, max_acc_name

def test_classifiers(name, classifier, vocabulary, emb_dictionary, embedding_size, dictionary, domains_centroid_max_min_vectors_dict, max_accuracy, num_test_file, d_id):
    print("[Test]")
    file_name = str(d_id) + '_' + str(name) + '_' + str(max_accuracy) + '_test_answers.tsv'
    answer_file = os.path.join(out_dir, file_name)
    test_files = get_files_and_domain(TEST_DIR, shuffle=False)
    # __import__('ipdb').set_trace()
    bar = None
    if num_test_file == -1:
        bar = tqdm.trange(len(test_files))
    else:
        bar = tqdm.trange(len(test_files[:num_test_file]))

    for step in bar:
        t_file, _ = test_files[step]
        base_name = os.path.basename(t_file)
        t_id = base_name.split('.')[0]
        doc_test_file, _ = fetch_docs_with_labels([(t_file,_)], 'I_DO_NOT_EXIST', load_bar=False, caching=False)
        label = [None]
        args = [doc_test_file, label, vocabulary, emb_dictionary, embedding_size, dictionary, domains_centroid_max_min_vectors_dict]
        test, _ = docs2vec(docs2vec_words_peculiarity_weighted_centroid, *args)
        prediction = classifier.predict(test)
        add_answer(t_id, prediction[0], answer_file)

    # __import__('ipdb').set_trace()




def docs2vec(to_vector, *args):
    training_set, labels = to_vector(*args)
    return training_set, labels


def docs2vec_words_peculiarity_weighted_centroid(docs_training_set, labels, vocabulary, emb_dictionary, embedding_size, dictionary, domains_centroid_max_min_vectors_dict, weighted=False):
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
        # print('Label: \n{}'.format(label))
        # print('Words: \n{}'.format(words))
        # print('Peculiarities: \n{}'.format([float(pec) / (10**9) for pec in pecurliarities]))
        # input("Press ENTER to continue.")
        embedded_words = dict()
        try:
            embedded_words = [emb_dictionary[word] for word in words if vocabulary[word]]
        except KeyError as e:
            print("not in emb_dict - should never be printed")
        try:
            centroid = None
            if weighted: centroid = mean_vector(embedded_words, pecurliarities, embedding_size)
            else: centroid = avg_vector(embedded_words, embedding_size)
            max_vec = max_vector(embedded_words, embedding_size)
            min_vec = min_vector(embedded_words, embedding_size)
            # sorted cause I want it to be same order for all features
            cos_sim_feat_vec = []
            for domain, d_vectors in sorted(domains_centroid_max_min_vectors_dict.items()):
                d_centroid, d_max_vec, d_min_vec, d_avg_vec = d_vectors
                # __import__('ipdb').set_trace()
                centroid_sim = None
                if weighted: centroid_sim = cosine_similarity(d_centroid, centroid)
                else: centroid_sim = cosine_similarity(d_avg_vec, centroid)
                max_sim =  cosine_similarity(d_max_vec, max_vec)
                min_sim =  cosine_similarity(d_min_vec, min_vec)
                cos_sim_feat_vec += [centroid_sim, max_sim, min_sim]

            training_set.append(centroid + max_vec + min_vec + cos_sim_feat_vec)
            # append label only if append training example
            new_labels.append(label)
        except ZeroDivisionError as e:
            continue
        # print(len(training_set), len(new_labels))
        assert len(training_set) == len(new_labels), "INVALID OUTPUT"
    return training_set, new_labels


def peculiar_doc_words(doc_vocab, vocabulary):
    tot_occ_doc = sum(doc_vocab.values())
    tot_occ_dataset = sum(vocabulary.values())
    peculiar_word = Counter()
    for word in doc_vocab:
        occur = doc_vocab[word]
        # assert that doc_vocab is contained in vocabulary
        if not vocabulary[word]:
            continue
        freq_in_doc = occur / tot_occ_doc
        freq_in_dataset = vocabulary[word] / tot_occ_dataset
        freq_ratio = freq_in_doc / freq_in_dataset
        precision = 10**9
        peculiarity = int(freq_ratio * precision)
        peculiar_word[word] = peculiarity
    pecurliar_words = [word for word in peculiar_word]
    pecurliarities = [peculiar_word[word] for word in peculiar_word]
    return pecurliar_words, pecurliarities


def cosine_similarity(vec1, vec2):
    vec1 = np.asanyarray(vec1)
    vec2 = np.asanyarray(vec2)
    sim = np.sum(vec1*vec2) / (np.sqrt(np.sum(np.power(vec1, 2))) * np.sqrt(np.sum(np.power(vec2, 2))))
    if math.isnan(sim):
        sim = 0
    return sim

def domains_centroid_max_min_vectors(ds_vocabs, vocabulary, embedding_size, emb_dictionary, caching_directory, caching=True):
    domains_centroid_max_min_vectors_dict = dict()
    bak = os.path.join(caching_directory, "domains_centroid_max_min_vectors_dict" + ".pickle")
    if caching and os.path.exists(bak):
        print('Loading domains_centroid_max_min_vectors_dict from previous same execution.')
        with LogTime('Loading domains_centroid_max_min_vectors_dict'):
            with open(bak, 'rb') as f:
                domains_centroid_max_min_vectors_dict = pickle.load(f)
    else:
        for domain, domain_vocabulary in ds_vocabs.items():
            assert is_sub_vocab(domain_vocabulary, vocabulary)
            embedded_words = dict()
            # __import__('ipdb').set_trace()
            peculiarities_domain = [domain_vocabulary[word] for word in domain_vocabulary]
            peculiarities_dataset = [vocabulary[word] for word in vocabulary]
            peculiarities_ratio = [float(domain_p)/dataset_p for domain_p, dataset_p in zip(peculiarities_domain, peculiarities_dataset)]
            embedded_domain_words = [emb_dictionary[word] for word in domain_vocabulary]
            d_centroid = mean_vector(embedded_domain_words, peculiarities_ratio, embedding_size)
            d_max_vec = max_vector(embedded_words, embedding_size)
            d_min_vec = min_vector(embedded_words, embedding_size)
            d_avg_vec = avg_vector(embedded_words, embedding_size)
            domains_centroid_max_min_vectors_dict[domain] = (d_centroid, d_max_vec, d_min_vec, d_avg_vec)

        if caching and not os.path.exists(bak):
            with LogTime('Caching domains_centroid_max_min_vectors_dict'):
                with open(os.path.join(caching_directory, "domains_centroid_max_min_vectors_dict" + ".pickle"), 'wb') as f:
                    pickle.dump(domains_centroid_max_min_vectors_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return domains_centroid_max_min_vectors_dict


def is_sub_vocab(sub_vocab, vocab):
    return all(sub_vocab[x] <= vocab[x] for x in sub_vocab)


def mean_vector(vectors, weights, embedding_size):
    # all vector must have same dimention
    sum_w = sum(weights)
    centroid = [0] * embedding_size
    for v_index, vector in enumerate(vectors):
        for dimention, coordinate in enumerate(vector):
            # __import__('ipdb').set_trace()
            # print('w: {}, c: {}'.format(weights[v_index], coordinate))
            centroid[dimention] += weights[v_index] * coordinate
    centroid = [ x / sum_w for x in centroid ]
    return centroid

# I pass embedding size cause vectors may empty
def pointwise_f_vector(vectors, embedding_size, f):
    pointwise_vector = [0] * embedding_size
    if len(vectors) == 0:
        print('[ATTENTION]: found empty doc -> pointwise_vector is 0...0')
        return pointwise_vector
    for dim in range(embedding_size):
        values = [float(vector[dim]) for vector in vectors]
        pointwise_vector[dim] = f(values)
    return pointwise_vector

def max_vector(vectors, embedding_size):
    return pointwise_f_vector(vectors, embedding_size, min)

def min_vector(vectors, embedding_size):
    return pointwise_f_vector(vectors, embedding_size, min)

def avg_vector(vectors, embedding_size):
    return pointwise_f_vector(vectors, embedding_size, mean)


def recover_execution_environment(exec_id):
    vocabulary = parse_vocab(path_of('vocab', exec_id, ext='.csv'))
    # with open(path_of('vocab', exec_id), 'rb') as f:
    #     vocabulary = pickle.load(f)
    # vector_file = path_of('vectors', exec_id)
    # vectors = parse_vectors(vector_file)
    with open(path_of('vectors', exec_id), 'rb') as f:
        vectors = pickle.load(f)
    dic_file = path_of('dict', exec_id, ext='.npy')
    dictionary = load_dictionary(dic_file)
    # with open(path_of('dict', exec_id), 'rb') as f:
    #     dictionary = pickle.load(f)
    inv_dic_file = path_of('inv_dict', exec_id, ext='.npy')
    inv_dictionary = load_dictionary(inv_dic_file)
    # with open(path_of('inv_dict', exec_id), 'rb') as f:
        # inv_dictionary = pickle.load(f)
    emb_dictionary = embeddings_dictionary(vectors, inv_dictionary)
    embedding_size = len(vectors[0])
    # execution = Execution(exec_id, embedding_size, vocabulary, vectors, dictionary, inv_dictionary, emb_dictionary)
    return vocabulary, vectors, dictionary, inv_dictionary, emb_dictionary, embedding_size


def fetch_docs_with_labels(training_files, caching_directory, load_bar=True, caching=True):
    docs_training_set = None
    labels = None
    bak1 = os.path.join(caching_directory, "word_training_set" + ".pickle")
    if caching and os.path.exists(bak1):
        print('Loading word_test_set and labels from previous same execution.')
        with LogTime('Loading docs_training_set and labels'):
            with open(bak1, 'rb') as f:
                docs_training_set = pickle.load(f)
            with open(os.path.join(caching_directory, "labels" + ".pickle"), 'rb') as f:
                labels = pickle.load(f)
    else:
        docs_training_set = []
        labels = []
        # print('fetch_docs_with_labels')
        bar = None
        if load_bar:
            bar = tqdm.trange(len(training_files))
        else:
            bar = range(len(training_files))
        for i in bar:
            f, domain = training_files[i]
            file_sentences = data_from_file(f) #, num_file_words)
            docs_training_set.append(file_sentences)
            labels.append(domain)

        if caching and not os.path.exists(bak1):
            with LogTime('Caching docs_training_set and labels'):
                with open(os.path.join(caching_directory, "word_training_set" + ".pickle"), 'wb') as f:
                    pickle.dump(docs_training_set, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(caching_directory, "labels" + ".pickle"), 'wb') as f:
                    pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    return docs_training_set, labels

def data_from_file(file_name):
    sentences = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            split = line.lower().strip().split()
            sentences.append(split)
    return sentences

# embeddings_dict[word] -> word_vector
def embeddings_dictionary(vectors, inv_dictionary):
    assert len(inv_dictionary) == len(vectors)
    embeddings_dict = dict()
    for index, vector in enumerate(vectors):
            # TODO check if our of one
        embeddings_dict[inv_dictionary[index]] = vector
    return embeddings_dict


def parse_vocab(vocab_file):
    vocab = Counter()
    with open(vocab_file) as f:
        for line in f:
            # print(line)
            word, occur = line.split(',')
            vocab[word] = int(occur)
    return vocab


def load_dictionary(dic_file):
    dictionary = np.load(dic_file).item()
    return  dictionary

def path_of(entity, exec_id, ext='.pickle'):
    path = os.path.join(log_input_dir, 'log/', entity, exec_id + ext)
    return path


# computes the dictionary: domain_name -> domain_vocabulary, of type str -> Counter
def domains_vocabularies(docs_training_set, labels, vocabulary, caching_directory, caching=True):
    bak = os.path.join(caching_directory, "ds_vocabs" + ".pickle")
    if caching and os.path.exists(bak):
        print('Loading domains_vocabularies from previous same execution.')
        with LogTime('Loading domains_vocabularies'):
            with open(bak, 'rb') as f:
                ds_vocabs = pickle.load(f)
    else:
        ds_vocabs = dict() # domain -> domain_vocabulary
        with LogTime('Domain Vocabularies'):
            domains_counter = Counter(labels)
            for domain in domains_counter:
                domain_vocabulary = Counter()
                print("Domain: {}, Docs: {}".format(domain, domains_counter[domain]))
                domain_doc_idx = [i for i, x in enumerate(labels) if x == domain]
                for idx in domain_doc_idx:
                    doc_sentences = docs_training_set[idx]
                    doc_vocab = docs_to_sub_vocab(doc_sentences, vocabulary)
                    domain_vocabulary += doc_vocab
                ds_vocabs[domain] = domain_vocabulary
        if caching and not os.path.exists(bak):
            # os.makedirs(caching_directory)
            with LogTime('Caching domains_vocabularies'):
                with open(bak, 'wb') as f:
                    pickle.dump(ds_vocabs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ds_vocabs

def docs_to_sub_vocab(doc_sentences, vocabulary):
    sub_vocab = Counter()
    for sentences in doc_sentences:
        for w in sentences:
            # excludes out_of_vocab words i.e., unk
            if vocabulary[w]:
                sub_vocab[w] += 1
    return sub_vocab

def add_answer(t_id, prediction, answer_file):
    answer = '{}\t{}\n'.format(t_id, prediction)
    log(answer_file, answer)

def get_domain_doc_stats():
    domain_doc_number = dict()
    for domain in os.listdir(TRAIN_DIR):
        if domain != '.DS_Store':
            doc_num = len([f for f in os.listdir(os.path.join(TRAIN_DIR, domain))])
            domain_doc_number[domain] = doc_num
    num_docs = sum(domain_doc_number.values())
    return num_docs, domain_doc_number


if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except IndexError as e:
        print("Pass execution id, num training file, num validation files and num test files as arg of {}".format(sys.argv[0]))
        sys.exit(1)
