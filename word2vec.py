import os
import math

import tensorflow as tf
import numpy as np
import tqdm
# from tensorboard.plugins import projector
from tensorflow.contrib.tensorboard.plugins import projector
from data_preprocessing import generate_batch, build_dataset, save_vectors, read_analogies, get_training_set_coverage, compute_training_set_cardinality, vocab_to_csv
from evaluation import evaluation

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import argparse

import time
# import ipdb

import pickle

# run on CPU
# comment this part if you want to run it on GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

### CONSTANTS ###
TRAIN_DIR      = "dataset/DATA/TRAIN"
VALID_DIR      = "dataset/DATA/DEV"
ANALOGIES_FILE = "dataset/eval/questions-words.txt"
STEP_CHECK     = 100000 # Every how many step to check and to log accuracy/loss.
# LOG_FILE       = "./log/log_to_plot.txt"
log_dirs       = ['log', 'log/tensorboard', 'log/executions', 'log/accuracy', 'log/loss', 'log/dict', 'log/inv_dict', 'log/vectors', 'log/vocab', 'log/caching', 'log/lr']

decay_step = 100000
decay_rate = 0.96

### MAIN {{{
def main ():
    ### CLI args ###
    cmdLineParser = argparse.ArgumentParser()
    cmdLineParser.add_argument("batch_size", type=int, help="Number of samples per batch.")
    cmdLineParser.add_argument("embedding_size", type=int, help="Dimension of the embedding vector.")
    cmdLineParser.add_argument("window_size", type=int, help="How many context words to consider left and right drom pivot word.")
    cmdLineParser.add_argument("neg_samples", type=int, help="Number of negative examples to sample for skipgram model.")
    cmdLineParser.add_argument("vocabulary_size", type=int, help="(At most) Number of known words.")
    cmdLineParser.add_argument("num_domain_words", type=int, help="Number of words for each domain.")
    cmdLineParser.add_argument("num_steps", type=int, help="number of training iterations")
    cmdLineParser.add_argument("learning_rate", type=float, help="base learning rate")
    cmdLineParser.add_argument('step_rate', nargs='*', default=[], help='[decay step, decay rate]')
    cmdLineParser.add_argument("--decay", dest="decay", action='store_true')
    cmdLineParser.add_argument("--linear_decay", dest="linear_decay", action='store_true')
    cmdLineParser.add_argument("--shuffle_data", dest="shuffle_data", action='store_true')
    cmdLineParser.add_argument("--catching", dest="catching", action='store_true')
    cmdLineArgs = cmdLineParser.parse_args()

    batch_size       = cmdLineArgs.batch_size
    embedding_size   = cmdLineArgs.embedding_size
    window_size      = cmdLineArgs.window_size
    neg_samples      = cmdLineArgs.neg_samples
    vocabulary_size  = cmdLineArgs.vocabulary_size
    num_domain_words = cmdLineArgs.num_domain_words
    num_steps        = cmdLineArgs.num_steps
    learning_rate    = cmdLineArgs.learning_rate
    decay = cmdLineArgs.decay
    linear_decay = cmdLineArgs.linear_decay
    step_rate = cmdLineArgs.step_rate
    shuffle_data = cmdLineArgs.shuffle_data
    catching = cmdLineArgs.catching
    print("DECAY: ", decay, step_rate)

    hyperparameters = [batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate, decay, linear_decay, step_rate, catching, shuffle_data]

    # Need for a 'sort-of' unique, ordered id to identify different executions in log.
    execution_id = int(time.time()*100)

    # Creates directory needed for log.
    for directory in log_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


    # Pre-exec log.
    log('./log/executions/' + str(execution_id) + '.txt', "Execution ID:" + str(execution_id) + ':' + str(hyperparameters) + '\n')

    # Execution
    start = time.time()
    final_relative_accuracy, acc_perc, avg_iteraz_sec, final_avg_loss, coverage_data = train(*hyperparameters, execution_id)
    stop  = time.time()

    decay_method = 'None'
    if decay:
        decay_method = 'Exponential Decay'
    if linear_decay:
        decay_method = 'Linear Decay'

    # Post-exec log, only executions carried out till completion.
    log('./log/executions/' + 'log' + '.txt', "Execution ID:" + str(execution_id) + ':' + str(hyperparameters) + '\n')
    log('./log/executions/' + 'log' + '.txt', "Acc: " + str(final_relative_accuracy) + ", Acc%: " + str(acc_perc) + "%, It/s: " + str(avg_iteraz_sec) + ", Loss: " + str(final_avg_loss) + ", Decay: " + decay_method + '\n')

    # training_pairs, used_training_pairs, coverage, coverage_unk = coverage_data
    training_pairs, epoch, coverage, training_set_cardinality = coverage_data


    log('./log/executions/' + 'log' + '.txt', "training_pairs: " + str(training_pairs) + ", epoch: " + str(epoch) + ", coverage: " + str(coverage) + "%, Training_set_card: " + str(training_set_cardinality)+ '\n')
    log('./log/executions/' + 'log' + '.txt', "----Completion time (min): " + str(int((stop-start)/60))+'\n')
### }}} END MAIN


def train(batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate, decay, linear_decay, step_rate, shuffle_data, catching, execution_id):

    hyperparameters = [batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate, decay, linear_decay, step_rate, shuffle_data, catching, execution_id]
    print('Current Execution: ', hyperparameters)



    dictionary = None
    reverse_dictionary = None
    vocabulary = None
    coverage_data = None
    data = None
    cardinality = None

    is_shuffle = "no_shuffle"
    if shuffle_data: is_shuffle = "shuffle"
    dw = str(num_domain_words)
    if num_domain_words == -1: dw = 'all'

    back_env = os.path.join('log/caching', is_shuffle, dw, str(vocabulary_size))

    if catching and os.path.exists(back_env):
        print('Loading env from previous execution with same hyperparameters.')
        start    = time.time()
        # Load environment from previous pickled execution
        with open(os.path.join(back_env, "dict" + ".pickle"), 'rb') as f:
            dictionary = pickle.load(f)
        with open(os.path.join(back_env, "inv_dict" + ".pickle"), 'rb') as f:
            reverse_dictionary = pickle.load(f)
        with open(os.path.join(back_env, "vocab" + ".pickle"), 'rb') as f:
            vocabulary = pickle.load(f)
        with open(os.path.join(back_env, "cvrg" + ".pickle"), 'rb') as f:
            coverage_data = pickle.load(f)
        with open(os.path.join(back_env, "data" + ".pickle"), 'rb') as f:
            data = pickle.load(f)
        with open(os.path.join(back_env, "crdn" + ".pickle"), 'rb') as f:
            cardinality = pickle.load(f)

        print('Env loading time: {}'.format(time.time() - start))

    else:
        # Compute environment and pickle them

        # load the training set
        start    = time.time()
        raw_data = None
        if shuffle_data:
            raw_data = read_shuffled_data(TRAIN_DIR, num_dataset_words=num_domain_words)
        else:
            raw_data = read_data(TRAIN_DIR, domain_words=num_domain_words)
        stop     = time.time()
        dur      = stop - start
        print('Raw Data size: ', len(raw_data), 'Time raw_data: ', dur)

        # Data stat to log
        num_sentences = len(raw_data)
        num_word = 0
        for s in raw_data:
            num_word += len(s)

        print("num_word: ", num_word)

        ### CREATE THE DATASET AND WORD-INT MAPPING ###
        start = time.time()
        data, dictionary, reverse_dictionary, vocabulary = build_dataset(raw_data, vocabulary_size, execution_id)
        stop  = time.time()
        dur   = stop - start
        print('Time bult_dataset: ', dur)

        del raw_data  # Hint to reduce memory.

        cardinality = compute_training_set_cardinality(window_size, data)
        coverage_data = get_training_set_coverage(batch_size, num_steps, cardinality)
        training_pairs, epoch, coverage, training_set_cardinality = coverage_data
        print("Training Set Cardinality: ", cardinality)
        print("Epoch: ", epoch)
        print("Coverage: {}%".format(coverage))

        # pickle
        # create dirs and save stuff

        if not os.path.exists(back_env):
            # Backup data only if not already done
            os.makedirs(back_env)
            with open(os.path.join(back_env, "dict" + ".pickle"), 'wb') as f:
                pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(back_env, "inv_dict" + ".pickle"), 'wb') as f:
                pickle.dump(reverse_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(back_env, "vocab" + ".pickle"), 'wb') as f:
                pickle.dump(vocabulary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(back_env, "cvrg" + ".pickle"), 'wb') as f:
                pickle.dump(coverage_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(back_env, "data" + ".pickle"), 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(back_env, "crdn" + ".pickle"), 'wb') as f:
                pickle.dump(cardinality, f, protocol=pickle.HIGHEST_PROTOCOL)

    # dump env for backward compatibility and for tracebility
    vocab_to_csv(vocabulary, vocabulary_size, execution_id)
    dict_file     = os.path.join("./log/dict", str(execution_id))
    np.save(dict_file + '.npy', dictionary)
    inv_dict_file = os.path.join("./log/inv_dict", str(execution_id))
    np.save(inv_dict_file + '.npy', reverse_dictionary)


    # the portion of the training set used for data evaluation
    valid_size     = 16  # Random set of words to evaluate similarity on.
    valid_window   = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    # read the question file for the Analogical Reasoning evaluation
    questions = read_analogies(ANALOGIES_FILE, dictionary)

    ### MODEL DEFINITION ###
    # with tf.device('/device:GPU:0'): # uncomment and indent all
    graph = tf.Graph()
    eval  = None
    # needed if in case of restoring env
    # ipdb.set_trace()
    training_pairs, epoch, coverage, training_set_cardinality = coverage_data
    with graph.as_default():
        # Define input data tensors.
        with tf.name_scope('inputs'):
            train_inputs  = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels  = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        ### FILL HERE ###{{{

        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        selector = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_w = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_b  = tf.Variable(tf.zeros([vocabulary_size]))

        ### }}}

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=nce_w,
                biases=nce_b,
                labels=train_labels,
                inputs=selector,
                num_sampled=neg_samples,
                num_classes=vocabulary_size))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

        with tf.name_scope('optimizer'):
            decay_learning_rate = None
            if not decay and not linear_decay:
                # constant learning rate
                    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            else:
                # decay lr
                global_step = tf.Variable(0, trainable=False)
                intitial_learning_rate = learning_rate
                decay_step = int(step_rate[0])  #100000
                decay_rate = float(step_rate[1])  #0.96
                if decay:
                    decay_learning_rate = tf.train.exponential_decay(intitial_learning_rate, global_step, decay_step, decay_rate, staircase=True)
                elif linear_decay:
                    decay_learning_rate = tf.train.inverse_time_decay(intitial_learning_rate, global_step, decay_step, decay_rate, staircase=True)
                    # decay_learning_rate = tf.train.noisy_linear_cosine_decay(intitial_learning_rate, global_step, decay_step, decay_rate)

                # decay_learning_rate = tf.train.cosine_decay(intitial_learning_rate, global_step, decay_step, name=None)

                optimizer = tf.train.GradientDescentOptimizer(decay_learning_rate).minimize(loss, global_step=global_step)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        # Merge all summaries.
        merged = tf.summary.merge_all()

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Create a saver.
        saver = tf.train.Saver()

        # evaluation graph
        eval = evaluation(normalized_embeddings, dictionary, questions)

    tb_dir = os.path.join(os.getcwd(), 'log/tensorboard', str(execution_id))
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)


    ### TRAINING ###
    # Step 5: Begin training.
    with tf.Session(graph=graph) as session:
    #, config=tf.ConfigProto(log_device_placement=True)) as session:

        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(tb_dir, session.graph)
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        loss_over_time = []
        curr_sentence = 0
        curr_word =0
        curr_context_word = 0
        it_start = time.time()
        learning_rate_over_time = []
        local_max_acc = 0
        local_min_acc = 0
        last_max_update = 0
        max_update_delta = 0
        max_update_period = 0
        bar = tqdm.trange(num_steps)
        early_stop = False
        for step in bar:
            batch_inputs, batch_labels, cs, cw, ccw = generate_batch(batch_size, curr_sentence, curr_word, curr_context_word, window_size, data)
            curr_sentence = cs
            curr_word = cw
            curr_context_word = ccw

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op
            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict={train_inputs: batch_inputs, train_labels: batch_labels},
                run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            if early_stop:
                writer.add_run_metadata(run_metadata, 'step%d' % step)
                print('Suspected to be nearby local max, early stopping.')
                break
            # Add metadata to visualize the graph for the last run.
            if step % STEP_CHECK == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)
            if step % STEP_CHECK is 0:
                # print('Current Execution: ', hyperparameters)
                eval.eval(session)
                if step != 0:
                    print("avg loss: "+str(average_loss/step))
                    loss_over_time.append(average_loss/step)
                initial_acc = eval.accuracy_log[0]
                curr_acc = eval.accuracy_log[-1]

                # Stats over accuracy, to have an estimation at runtime if it get stuck in a local max.
                last_max_update += 1
                if curr_acc > local_max_acc:
                    local_max_acc = curr_acc
                    local_min_acc = local_max_acc
                    if max_update_period < last_max_update:
                        # max { incremental ratio of period between two accuracy local max }
                        if max_update_period and max_update_delta < last_max_update / max_update_period:
                            max_update_delta = last_max_update / max_update_period
                        max_update_period = last_max_update


                    # Stop in accuracy local max if nearby the end, not if testing
                    next_max_accuracy_expected_iteration = step + (last_max_update * max_update_delta * STEP_CHECK)
                    if next_max_accuracy_expected_iteration > num_steps: # and num_steps > 10**6:
                        early_stop = True
                    else:
                        print('Not ready for early stopping, next max foreseen accuracy around iteration: {} < {}'.format(next_max_accuracy_expected_iteration, num_steps))
                    last_max_update = 0
                if curr_acc < local_min_acc:
                    local_min_acc = curr_acc

                # print("accuracy gain: "+str(curr_acc - initial_acc))
                print("HP: ", hyperparameters)
                print("Max_accuracy: ", local_max_acc)
                print("Latest max update: {} * {} it ago".format(last_max_update, STEP_CHECK))
                print("Max Update Period: ", max_update_period)
                print("Max Update Delta: ", max_update_delta)
                print("Min_local_accuracy: ", local_min_acc)
                print("Training Set Cardinality: ", cardinality)
                print("Epoch: ", epoch)
                print("Coverage: {}%".format(coverage))
                if decay or linear_decay:
                    if step % decay_step == 0:
                        learning_rate_over_time.append(float(decay_learning_rate.eval()))
                    if decay:
                        print("EXP Decay lr: ", decay_learning_rate.eval())
                    if linear_decay:
                        print("LIN Decay lr: ", decay_learning_rate.eval())

        it_stop = time.time()
        final_embeddings = normalized_embeddings.eval()

        # my function but is commented cause works only using gui
        eval.plot()

        final_relative_accuracy = eval.accuracy_log[-1]
        num_questions = eval.questions.shape[0]

        avg_iteraz_sec = num_steps / (it_stop - it_start)

        final_avg_loss = loss_over_time[len(loss_over_time) - 1]
        acc_perc = final_relative_accuracy * 100.0 / num_questions
        log_loss(execution_id, loss_over_time)
        log_accuracy(execution_id, eval.accuracy_log)
        if decay or linear_decay:
            log_learning_rate(execution_id, learning_rate_over_time)

        ### SAVE VECTORS ###
        if final_relative_accuracy > 7000:
            save_vectors(final_embeddings, execution_id)

        # Write corresponding labels for the embeddings.
        with open(os.path.join(tb_dir, 'metadata.tsv'), 'w') as f:
            for i in range(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        # Save the model for checkpoints
        saver.save(session, os.path.join(tb_dir, 'model.ckpt'))

        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(tb_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

    writer.close()

    # pdb.set_trace()

    return final_relative_accuracy, acc_perc, avg_iteraz_sec, final_avg_loss, coverage_data

### READ THE TEXT FILES ###
# Read the data into a list of strings.
# the domain_words parameters limits the number of words to be loaded per domain
def read_data(directory, domain_words=-1):
    data = []
    for domain in os.listdir(directory):
    #for dirpath, dnames, fnames in os.walk(directory):
        limit = domain_words
        # Compatibility with macOS
        if domain == ".DS_Store":
            continue
        for f in os.listdir(os.path.join(directory, domain)):
            if f.endswith(".txt"):
                with open(os.path.join(directory, domain, f)) as file:
                    # sentences = []
                    for line in file.readlines():
                        split = line.lower().strip().split()
                        if limit > 0 and limit - len(split) < 0:
                            split = split[:limit]
                        elif limit != -1:
                            limit -= len(split)
                        if limit >= 0 or limit == -1:
                            data.append(split)
                    # data.append(sentences)
    return data


def read_shuffled_data(directory, num_dataset_words=-1):
    files = get_files_and_domain(directory, shuffle=True)
    data = []
    limit = num_dataset_words
    for file_index in tqdm.trange(len(files)):
        f, _ = files[file_index]
        with open(f) as file:
            # sentences = []
            for line in file.readlines():
                split = line.lower().strip().split()
                if limit > 0 and limit - len(split) < 0:
                    split = split[:limit]
                elif limit != -1:
                    limit -= len(split)
                if limit >= 0 or limit == -1:
                    data.append(split)
    return data

def get_files_and_domain(directory, shuffle=False):
    files = []
    for domain in os.listdir(directory):
        # Compatibility with macOS
        if domain == ".DS_Store":
            continue
        for f in os.listdir(os.path.join(directory, domain)):
            file_path = os.path.join(directory, domain, f)
            if f.endswith(".txt"):
                files.append((file_path, domain))

    files = np.asarray(files, dtype=str)
    if shuffle:
        # start = time.time()
        np.random.shuffle(files)
        # stop = time.time()
        # print('Shuffle: ', stop-start)
    return files

# appends text_to_log to log_file
def log(log_file, text_to_log):
    with open(log_file, "a") as log:
        log.write(text_to_log)

def log_accuracy(execution_id, accuracy):
    file_name = os.path.join("./log/accuracy", str(execution_id) + ".csv")
    np.savetxt(file_name, np.asarray(accuracy), delimiter=',')
    # with open(file_name, "w") as acc_log:
        # acc_log.write(accuracy)

def log_loss(execution_id, loss):
    file_name = os.path.join("./log/loss", str(execution_id) + ".csv")
    np.savetxt(file_name, np.asarray(loss), delimiter=',')
    # with open(file_name, "w") as loss_log:
        # loss_log.write(loss)

def log_learning_rate(execution_id, rate):
    file_name = os.path.join("./log/lr", str(execution_id) + ".csv")
    np.savetxt(file_name, np.asarray(rate), delimiter=',')


if __name__ == '__main__':
    main()
