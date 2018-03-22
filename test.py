import numpy as np
import tensorflow as tf
# from data_preprocessing import get_stopwords
# from data_preprocessing import build_dataset
from word2vec import read_data
# from word2vec import TRAIN_DIR
from data_preprocessing import build_dataset
from data_preprocessing import generate_batch

VOCABULARY_SIZE = 50000

print('TEST')


# WORDS = ['the', 'barometric', 'formula,', 'sometimes', 'called', 'the',
#          'exponential', 'atmosphere', 'or', 'isothermal', 'atmosphere,', 'is',
#          'a', 'formula', 'used', 'to', 'model', 'how', 'the', 'pressure', '(or',
#          'density)', 'of', 'the', 'air', 'changes', 'with', 'altitude.', 'there',
#          'are', 'two', 'different', 'equations', 'for', 'computing', 'pressure',
#          'at', 'various', 'height', 'regimes', 'below', '86', 'km', '(or',
#          '278,400', 'feet).', 'the', 'first', 'equation', 'is', 'used', 'when',
#          'the', 'value', 'of', 'standard', 'temperature', 'lapse', 'rate', 'is',
#          'not', 'equal', 'to', 'zero;', 'the', 'second', 'equation', 'is',
#          'used', 'whenl', 'standard', 'temperature', 'lapse', 'rate', 'equals',
#          'zero.']

# DATA, dictionary, reverse_dictionary = build_dataset(WORDS, VOCABULARY_SIZE)

print('raw_data')
raw_data = read_data("dataset/DATA/TRAIN", domain_words=1)
data, dictionary, reverse_dictionary = build_dataset(raw_data, VOCABULARY_SIZE)
print('done raw_data')

# print(WORDS[10])
# print(WORDS[15])
# print(len(WORDS))
# for w in WORDS[10:15]:
#     print('hi' + w)

# DATA = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# x_train, y_train = generate_batch(1000, 1, 5, DATA)
x_train, y_train = generate_batch(1000, 1, 5, raw_data)
print('generated batch')

# print("DATA:")
# print(DATA)
# print("x_train:")
# print(x_train)
# print("y_train:")
# print(y_train)

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print('x_train shape', x_train.shape, '\n'
      'y_train_shape', y_train.shape)

# placeholders for x_train and y_train
x = tf.placeholder(tf.int32, shape=(None,))
#tf.float32 (if you use ONE-HOT encodings), shape=(None, vocab_size))
y = tf.placeholder(tf.int32, shape=(None,))
#tf.float32, shape=(None, vocab_size))
embedding_dim = 10

# bias is not necessary (just for this exercise)
W1 = tf.Variable(tf.random_normal([VOCABULARY_SIZE, embedding_dim]))
b1 = tf.Variable(tf.random_normal([embedding_dim]))

hidden_representation = tf.add(tf.nn.embedding_lookup(W1, x), b1)
#tf.add(tf.matmul(x, W1), b1)

W2 = tf.Variable(tf.random_normal([embedding_dim, VOCABULARY_SIZE]))
b2 = tf.Variable(tf.random_normal([VOCABULARY_SIZE]))
output = tf.add(tf.matmul(hidden_representation, W2), b2)
# prediction = tf.nn.softmax(output)


# define the loss function:
cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(y, output)
# IN-HOUSE, WITH THE PROBLEM THAT LOG WITH 0 VALUES WILL NOT BE SMOOTHED:
# tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), axis=[1]))

# define the training step:
train_step = tf.train.AdamOptimizer(2).minimize(cross_entropy_loss)
# (learning rate 2 (high...) for the optimizer)
# alternative: GradientDescentOptimizer(0.1)

n_iterations = 10

### TRAINING ###
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #bar = tqdm(range(n_iterations))
    # train for n_iterations iterations
    for k in range(n_iterations): #in bar:
        _, loss = sess.run([train_step, cross_entropy_loss], feed_dict={x: x_train, y: y_train})
        print('iteration', k, 'loss is : ', loss)
#bar.set_postfix({'iteration':k, 'loss':loss})
    vectors = sess.run(W1 + b1)
    print("first vector:", vectors[0])

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def cosine_similarity(vec1, vec2):
    return np.sum(vec1*vec2) / (np.sqrt(np.sum(np.power(vec1, 2))) * np.sqrt(np.sum(np.power(vec2, 2))))

def find_closest(word_index, vectors): # to act like positive infinity
    min_dist = 10000
    query_vector = vectors[word_index]
    counter = {}
    for index, vector in enumerate(vectors):
        dist = cosine_similarity(vector, query_vector)
        if dist < min_dist:
            list = counter.get(dist, [])
            list.append(index)
            counter[dist] = list
    return counter

print('barometric=', vectors[dictionary['barometric']])
closest = find_closest(dictionary['barometric'], vectors)

for w in sorted(closest.keys(), reverse=True):
    print(w, list(map(lambda x: reverse_dictionary[x], closest[w])))
