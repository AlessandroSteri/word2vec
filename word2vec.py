import os
import math

import tensorflow as tf
import numpy as np
import tqdm
from tensorboard.plugins import projector
from data_preprocessing import generate_batch, build_dataset, save_vectors, read_analogies
from evaluation import evaluation

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import pdb
import time

# run on CPU
# comment this part if you want to run it on GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

### PARAMETERS ###
'''
# BATCH_SIZE -> [ 32, 128, 256, 512, 1024, 2048 ]
BATCH_SIZE      = 128*2 #*2*2*2 #*2 #Number of samples per batch
EMBEDDING_SIZE   = 128 # Dimension of the embedding vector.
WINDOW_SIZE      = 2  # How many words to consider left and right.
NEG_SAMPLES      = 64  # Number of negative examples to sample.
VOCABULARY_SIZE  = 15000 #0 #The most N word to consider in the dictionary

# TODO my parameter
NUM_DOMAIN_WORDS = 50000#160*(10**6) # 0 #00 # was 1000
NUM_STEPS        = 20000 #0 #0 #0
#
# hyper_params   = [0.5, 1.0, 2.0]
LEARNING_RATE    = 0.5
 '''

TRAIN_DIR        = "dataset/DATA/TRAIN"
VALID_DIR        = "dataset/DATA/DEV"
TMP_DIR          = "/tmp/"
ANALOGIES_FILE   = "dataset/eval/questions-words.txt"


STEP_CHECK       = 10000
LOG_FILE = "./log/log_to_plot.txt"

### MAIN {{{
def main ():
    # Need a 'sort-of' unique id for runs
    RUN_ID = int(time.time())

    # defaults
    batch_size       = 128*2 #*2*2*2 #*2 #Number of samples per batch
    embedding_size   = 128 # Dimension of the embedding vector.
    window_size      = 2  # How many words to consider left and right.
    neg_samples      = 64  # Number of negative examples to sample.
    vocabulary_size  = 15000 #0 #The most N word to consider in the dictionary
    num_domain_words = 50000
    num_steps        = 20000
    learning_rate    = 0.5


    # log
    hyperparameters = [batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, STEP_CHECK, learning_rate]
    log('./log/times/' + str(RUN_ID) + '.txt', "Run ID:" + str(RUN_ID) + ':' + str(hyperparameters) + '\n')


    # batch_size
    for param in [32, 128*4, 128*16]:
        start = time.time()
        train(param, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate)
        stop  = time.time()
        log('./log/times/' + str(RUN_ID) + '.txt', "BATCH_SIZE: " + str(param) + " Completion time (min): " + str(int((stop-start)/60))+'\n')


    # embedding_size
    for param in [150, 280, 400]:
        start = time.time()
        train(batch_size, param, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate)
        stop  = time.time()
        log('./log/times/' + str(RUN_ID) + '.txt', "EMBEDDING_SIZE: " + str(param) + " Completion time (min): " + str(int((stop-start)/60))+'\n')


    # window_size
    for param in [1, 3, 4]:
        start = time.time()
        train(batch_size, embedding_size, param, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate)
        stop  = time.time()
        log('./log/times/' + str(RUN_ID) + '.txt', "WINDOW_SIZE: " + str(param) + " Completion time (min): " + str(int((stop-start)/60))+'\n')


    # neg_sample
    for param in [1, 3, 5]:
        start = time.time()
        train(batch_size, embedding_size, window_size, param, vocabulary_size, num_domain_words, num_steps, learning_rate)
        stop  = time.time()
        log('./log/times/' + str(RUN_ID) + '.txt', "NEG_SAMPLE: " + str(param) + " Completion time (min): " + str(int((stop-start)/60))+'\n')


    # vocabulary_size
    for param in [20000, 50000, 100000]:
        start = time.time()
        train(batch_size, embedding_size, window_size, neg_samples, param, num_domain_words, num_steps, learning_rate)
        stop  = time.time()
        log('./log/times/' + str(RUN_ID) + '.txt', "VOCABULARY_SIZE: " + str(param) + " Completion time (min): " + str(int((stop-start)/60))+'\n')


    # num_domain_words
    for param in [200*(10**3), 800*(10**3), 1600*(10**3)]:
        start = time.time()
        train(batch_size, embedding_size, window_size, neg_samples, vocabulary_size, param, num_steps, learning_rate)
        stop  = time.time()
        log('./log/times/' + str(RUN_ID) + '.txt', "NUM_DOMAIN_WORDS: " + str(param) + " Completion time (min): " + str(int((stop-start)/60))+'\n')


    # num_step
    for param in [200*(10**3), 200*(10**3), 200*(10**3)]:
        start = time.time()
        train(batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, param, learning_rate)
        stop  = time.time()
        log('./log/times/' + str(RUN_ID) + '.txt', "NUM_STEPS: " + str(param) + " Completion time (min): " + str(int((stop-start)/60))+'\n')


    # learning_rate
    for param in [20000, 50000, 100000]:
        start = time.time()
        train(batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, param)
        stop  = time.time()
        log('./log/times/' + str(RUN_ID) + '.txt', "VOCABULARY_SIZE: " + str(param) + " Completion time (min): " + str(int((stop-start)/60))+'\n')


def train(batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate):
    # Need a 'sort-of' unique id for execution
    EXECUTION_ID    = int(time.time()*100)
    hyperparameters = [batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, STEP_CHECK, learning_rate]
    print(hyperparameters)

    # minimal log to save different executions and select hyperparameters
    log(LOG_FILE, "ExecutionID:" + str(EXECUTION_ID) + ':')
    log(LOG_FILE, str(hyperparameters) + '\n')

    # load the training set
    raw_data = read_data(TRAIN_DIR, domain_words=num_domain_words)
    print('Data size: ', len(raw_data))
    # the portion of the training set used for data evaluation

    valid_size     = 16  # Random set of words to evaluate similarity on.
    valid_window   = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)


    ### CREATE THE DATASET AND WORD-INT MAPPING ###
    data, dictionary, reverse_dictionary = build_dataset(raw_data, vocabulary_size, EXECUTION_ID)

    # dump dictionaries
    dict_file = os.path.join("./log/dict", str(EXECUTION_ID))
    np.save(dict_file + '.npy', dictionary)
    inv_dict_file = os.path.join("./log/inv_dict", str(EXECUTION_ID))
    np.save(inv_dict_file + '.npy', reverse_dictionary)

    del raw_data  # Hint to reduce memory.
    # read the question file for the Analogical Reasoning evaluation
    questions = read_analogies(ANALOGIES_FILE, dictionary)

    print('Total words occurencies: {}'.format(len(data)))


    ### MODEL DEFINITION ###

    graph = tf.Graph()
    eval  = None

    with graph.as_default():
        # Define input data tensors.
        with tf.name_scope('inputs'):
            train_inputs  = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels  = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        ### FILL HERE ###{{{

        ### }}}
        # TODO:                 taken from slides
        embeddings            = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # ^ was embeddings    = tf.Variable() #placeholder variable
        # emb_bias            = tf.Variable(tf.random_normal(EMBEDDING_SIZE)) #placeholder variable
        ### FILL HERE ###{{{
        # TODO:                 taken from slides
        # selects column from index instead of product
        hidden_representation = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        # TODO:       just inverted dimension acccording to google
        # W2        = tf.Variable(tf.random_normal([VOCABULARY_SIZE, EMBEDDING_SIZE]))
        # output    = tf.matmul(hidden_representation, W2)
        nce_biases  = tf.Variable(tf.zeros([vocabulary_size]))

        ### }}}

        with tf.name_scope('loss'):
            # TODO: taken from slides
            # was: loss = None ### FILL HERE ###
            # loss = tf.losses.sparse_softmax_cross_entropy(train_labels, output)
            loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=hidden_representation,
                num_sampled=neg_samples,
                num_classes=vocabulary_size))

            # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            # TODO: taken from slides
            # was: optimizer = None ###FILL HERE ###
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

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

    ### TRAINING ###

    # Step 5: Begin training.

    with tf.Session(graph=graph) as session:
    #, config=tf.ConfigProto(log_device_placement=True)) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(TMP_DIR, session.graph)
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        loss_over_time = []
        bar = tqdm.trange(num_steps) #tqdm(range(num_steps))
        # batch_size, curr_sentence, curr_word, curr_context_word, window_size, data):
        curr_sentence = 0
        curr_word =0
        curr_context_word = 0
        it_start = time.time()
        for step in bar:
            batch_inputs, batch_labels, cs, cw, ccw = generate_batch(batch_size, curr_sentence, curr_word, curr_context_word, window_size, data)
            curr_sentence = cs
            curr_word = cw
            curr_context_word = ccw
            ### {{{
            # for x, y in zip(batch_inputs, batch_labels):
                # print("[ {}, {} ]".format(reverse_dictionary[x],reverse_dictionary[y]))
            ### }}}

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
                eval.eval(session)
                print("avg loss: "+str(average_loss/step))
                loss_over_time.append(average_loss/step)
                initial_acc = eval.accuracy_log[0]
                curr_acc = eval.accuracy_log[len(eval.accuracy_log) - 1]
                print("accuracy gain: "+str(curr_acc - initial_acc))

        it_stop = time.time()
        final_embeddings = normalized_embeddings.eval()

        ### SAVE VECTORS ###

        save_vectors(final_embeddings, EXECUTION_ID)
        # my function but is commented cause works only using gui on mbp
        eval.plot()
        final_relative_accuracy = eval.accuracy_log[len(eval.accuracy_log) - 1]
        num_questions = eval.questions.shape[0]
        final_absolute_accuracy = final_relative_accuracy / (eval.questions.shape[0])

        avg_iteraz_sec = (it_stop - it_start) / num_steps


        final_avg_loss = loss_over_time[len(loss_over_time) - 1]
        log(LOG_FILE, str([final_relative_accuracy, num_questions, final_absolute_accuracy, final_avg_loss, avg_iteraz_sec])+ '\n')
        log_loss(EXECUTION_ID, loss_over_time)
        log_accuracy(EXECUTION_ID, eval.accuracy_log)
        # print('TYPE: {}, LEN: {}'.format(type(eval.questions), len(eval.questions))


        # Write corresponding labels for the embeddings.
        with open(TMP_DIR + 'metadata.tsv', 'w') as f:
            for i in range(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        # Save the model for checkpoints
        saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))

        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(TMP_DIR, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

    writer.close()

    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    # plot_only = 500
    # low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    # labels = [reverse_dictionary[i] for i in range(plot_only)]
    # plot_with_labels(low_dim_embs, labels, os.path.join('./', 'tsne.png'))
    # pdb.set_trace()

### }}} END MAIN


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
                        else:
                            limit -= len(split)
                        if limit >= 0 or limit == -1:
                            data.append(split)
                    # data.append(sentences)
    return data


# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
                    label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom'
                    )
        plt.savefig(filename)


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


if __name__ == '__main__':
    main()
