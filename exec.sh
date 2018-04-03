



####  batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate
# python word2vec.py 32 128 1 5 150 8000 2000 1 100000 0.99 --decay # test execution
# 3k tier
# python word2vec.py 32 200 2 200 50000 800000000000 6400000 1 100000 0.99 --decay # test execution
python word2vec.py 32 200 10 200 50000 800000000000 6400000 1 100000 0.99 --decay # test execution
python word2vec.py 32 200 2 200 50000 800000000000 12800000 1 100000 0.99 --decay # test execution
python word2vec.py 32 200 10 200 50000 800000000000 12800000 1 100000 0.99 --decay # test execution

# best of till now
# python word2vec.py 32 200 1 200 100000 800000000000 400000 1 # 474 32min
# python word2vec.py 32 250 1 200 100000 800000000000 400000 1 # 470 31min

# todo imac, after plot accuracy of learning rate so to understand where to decay
# explore 300k words
# python word2vec.py 32 250 1 300 300000 800000000000 400000 1 #
# python word2vec.py 32 300 1 300 300000 800000000000 400000 1 #
# python word2vec.py 32 300 1 300 300000 800000000000 800000 1 #
#
#
# # explore learning rate
# python word2vec.py 32 200 1 200 100000 800000000000 800000 1 #
# python word2vec.py 32 200 1 200 100000 800000000000 1600000 1 #
# python word2vec.py 32 200 1 200 100000 800000000000 3200000 1 #
#
# python word2vec.py 32 200 1 200 100000 800000000000 800000 0.5 #
# python word2vec.py 32 200 1 200 100000 800000000000 1600000 0.5 #
# python word2vec.py 32 200 1 200 100000 800000000000 3200000 0.5 #
