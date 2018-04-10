

# Execution ID:152272740476:[32, 200, 2, 200, 50000, 800000000000, 12800000, 1.0, True, ['100000', '0.99']]
# Acc: 3820, Acc%: 30.007855459544384%, It/s: 437.9046109341637, Loss: 4.509408881700778
# training_pairs: 5607988051, used_training_pairs: 409600000, coverage: 230.87440722530926%, coverage_unk: 3160.9885668975644%, NumWord: 177412475
# ----Completion time (min): 509
# python word2vec.py 32 200 2 200 50000 800000000000 12800000 1 100000 0.99 --decay #

####  batch_size, embedding_size, window_size, neg_samples, vocabulary_size, num_domain_words, num_steps, learning_rate
# python word2vec.py 32 128 1 5 150 80000 2000 1 100000 0.99 --decay # test execution

# python word2vec.py 32 200 2 200 250 80000 200000 1 10000 0.5 --linear_decay #
# python word2vec.py 32 200 2 200 50000 800000000000 12000000 1 4000000 0.5 --linear_decay #
python word2vec.py 256 128 10 20 70000 -1 24000000 1 --shuffle_data #
#lr should end up to be around 0.07, if double iterations then final lr 0.005  maybe ry linear decay

