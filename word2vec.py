import os

import tensorflow as tf
import numpy as np
import tqdm
from tensorboard.plugins import projector
from data_preprocessing import generate_batch, build_dataset, save_vectors, read_analogies
from evaluation import evaluation


# run on CPU
# comment this part if you want to run it on GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

### PARAMETERS ###

BATCH_SIZE = 32 #Number of samples per batch
EMBEDDING_SIZE = 128 # Dimension of the embedding vector.
WINDOW_SIZE = 1  # How many words to consider left and right.
NEG_SAMPLES = 64  # Number of negative examples to sample.
VOCABULARY_SIZE = 50000 #The most N word to consider in the dictionary

# TODO my parameter
NUM_DOMAIN_WORDS = 10000 # was 1000

TRAIN_DIR = "dataset/DATA/TRAIN"
VALID_DIR = "dataset/DATA/DEV"
TMP_DIR = "/tmp/"
ANALOGIES_FILE = "dataset/eval/questions-words.txt"


### READ THE TEXT FILES ###

# Read the data into a list of strings.
# the domain_words parameters limits the number of words to be loaded per domain
def read_data(directory, domain_words=-1):
    data = []
    for domain in os.listdir(directory):
    #for dirpath, dnames, fnames in os.walk(directory):
        limit = domain_words
        for f in os.listdir(os.path.join(directory, domain)):
            if f.endswith(".txt"):
                with open(os.path.join(directory, domain, f)) as file:
                    for line in file.readlines():
                        split = line.lower().strip().split()
                        if limit > 0 and limit - len(split) < 0:
                            split = split[:limit]
                        else:
                            limit -= len(split)
                        if limit >= 0 or limit == -1:
                            data += split
    return data

# load the training set
raw_data = read_data(TRAIN_DIR, domain_words=NUM_DOMAIN_WORDS)
print('Data size', len(raw_data))
# the portion of the training set used for data evaluation
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


### CREATE THE DATASET AND WORD-INT MAPPING ###

data, dictionary, reverse_dictionary = build_dataset(raw_data, VOCABULARY_SIZE)
del raw_data  # Hint to reduce memory.
# read the question file for the Analogical Reasoning evaluation
questions = read_analogies(ANALOGIES_FILE, dictionary)

### MODEL DEFINITION ###

graph = tf.Graph()
eval = None

with graph.as_default():
    # Define input data tensors.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE]) #, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    ### FILL HERE ###{{{

    ### }}}
    # TODO: taken from slides
    embeddings = tf.Variable(tf.random_normal([VOCABULARY_SIZE, EMBEDDING_SIZE]))
    # ^ was embeddings = tf.Variable() #placeholder variable
    # emb_bias = tf.Variable(tf.random_normal(EMBEDDING_SIZE)) #placeholder variable
    ### FILL HERE ###{{{
    # TODO: taken from slides
    # selects column from index instead of product
    hidden_representation = tf.nn.embedding_lookup(embeddings, train_inputs)
    W2 = tf.Variable(tf.random_normal([EMBEDDING_SIZE, VOCABULARY_SIZE]))
    output = tf.matmul(hidden_representation, W2)



    ### }}}

    with tf.name_scope('loss'):
        # TODO: taken from slides
        # was: loss = None ### FILL HERE ###
        loss = tf.losses.sparse_softmax_cross_entropy(train_labels, output)

        # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        # TODO: taken from slides
        # was: optimizer = None ###FILL HERE ###
        optimizer = tf.train.AdamOptimizer(2).minimize(loss)

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
num_steps = 1000

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(TMP_DIR, session.graph)
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    bar = tqdm.tqdm(range(num_steps))
    for step in bar:
        batch_inputs, batch_labels = generate_batch(BATCH_SIZE, step, WINDOW_SIZE, data)

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
        if step % 10000 == 0:
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
        if step % 10000 is 0:
            eval.eval(session)
            print("avg loss: "+str(average_loss/step))
    final_embeddings = normalized_embeddings.eval()

    ### SAVE VECTORS ###

    save_vectors(final_embeddings)

    # Write corresponding labels for the embeddings.
    with open(TMP_DIR + 'metadata.tsv', 'w') as f:
        for i in range(VOCABULARY_SIZE):
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
