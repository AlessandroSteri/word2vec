



####  batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate
# python word2vec.py 32 128 1 5 150 8000 2000 1 # test execution

# test neg sample
# python word2vec.py 32 128 1 10 15000 800000 400000 1 # 222 20min
# python word2vec.py 32 128 1 50 15000 800000 400000 1 # 344 21min
# python word2vec.py 32 128 1 100 15000 800000 400000 1 # 438 21min
# python word2vec.py 32 128 1 200 15000 800000 400000 1 # 460 23min
#
# # test embedding size in large vocab
# python word2vec.py 32 128 1 5 100000 800000000000 400000 1 # 174 28min
# python word2vec.py 32 200 1 5 100000 800000000000 400000 1 # 186 28min
# python word2vec.py 32 400 1 5 100000 800000000000 400000 1 # 194 28min
#
# # test embedding vs large vocab vs neg sample
# python word2vec.py 32 128 1 20 100000 800000000000 400000 1 # 422 28min
# python word2vec.py 32 200 1 50 100000 800000000000 400000 1 # 435 29min
# python word2vec.py 32 400 1 100 100000 800000000000 400000 1 # 348 29min

# scenario large vocab, test emb size and neg sampl, nb having large vocab and few test may make neg sample very effective
python word2vec.py 32 200 1 200 100000 800000000000 400000 1 #
python word2vec.py 32 200 1 400 100000 800000000000 400000 1 #
python word2vec.py 32 250 1 200 100000 800000000000 400000 1 #
python word2vec.py 32 250 1 400 100000 800000000000 400000 1 #
python word2vec.py 32 400 1 200 100000 800000000000 400000 1 #
python word2vec.py 32 400 1 400 100000 800000000000 400000 1 #
python word2vec.py 32 128 1 200 100000 800000000000 400000 1 #

# try double the batch to see if time doubles and then double the iteration and compare accuracy to time
python word2vec.py 32 128 1 200 100000 800000000000 400000 1 # same test as line 12 (460 23min but with large vocab and words)
python word2vec.py 64 128 1 200 100000 800000000000 400000 1 # double batch
python word2vec.py 32 128 1 200 100000 800000000000 800000 1 # double iterations


# learning rate vs iterations.......select best of before
# python word2vec.py 32 128 1 200 100000 800000000000 1600000 1 # same test as before but 4x iterations
# python word2vec.py 32 128 1 200 100000 800000000000 1600000 0.5 # same test as before but 4x iterations
