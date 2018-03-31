



####  batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate
python word2vec.py 32 128 1 5 150 8000 20000 1 # test execution
python word2vec.py 32 128 1 5 150 8000 20000 1 # test execution
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
