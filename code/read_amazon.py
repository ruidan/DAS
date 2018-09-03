import codecs
import operator
import numpy as np
import re
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from read import create_vocab

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def create_data(vocab, file_path, skip_top, skip_len, replace_non_vocab):
    data = []
    f = codecs.open(file_path, 'r', 'utf-8')
    num_hit, unk_hit, skip_top_hit, total = 0., 0., 0., 0.
    max_len = 0

    for line in f:
        word_indices = []
        words = line.split()
        if skip_len > 0 and len(words) > skip_len:
            continue

        for word in words:
            if bool(num_regex.match(word)):
                word_indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                word_ind = vocab[word]
                if skip_top > 0 and word_ind < skip_top + 3:
                    skip_top_hit += 1
                else:
                    word_indices.append(word_ind)
            else:
                if replace_non_vocab:
                    word_indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        if len(word_indices) > max_len:
            max_len = len(word_indices)

        data.append(word_indices)

    print('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%, <skip_top> hit rate: %.2f%%' \
                        % (100*num_hit/total, 100*unk_hit/total, 100*skip_top_hit/total))

    return np.array(data), max_len


def prepare_data(source_domain, target_domain, n_class, vocab_size=0, skip_len=0, skip_top=0, replace_non_vocab=1):

    file_list = ['../data/amazon/%s/pos.txt'%source_domain, 
                 '../data/amazon/%s/neg.txt'%source_domain,
                 '../data/amazon/%s/un_pos.txt'%source_domain, 
                 '../data/amazon/%s/un_neg.txt'%source_domain,
                 '../data/amazon/%s/pos.txt'%target_domain, 
                 '../data/amazon/%s/neg.txt'%target_domain,
                 '../data/amazon/%s/un_pos.txt'%target_domain, 
                 '../data/amazon/%s/un_neg.txt'%target_domain]

    vocab = create_vocab(file_list, vocab_size, skip_len)

    data_list = []
    overall_max_len = 0
    for f in file_list:
        data, max_len = create_data(vocab, f, skip_top, skip_len, replace_non_vocab)
        data_list.append(data)
        if max_len > overall_max_len:
            overall_max_len = max_len

    return vocab, data_list, overall_max_len


def get_data(dataset, source_domain, target_domain, n_class, vocab_size=0):
    vocab, data_list, overall_maxlen = prepare_data(source_domain, target_domain, n_class, vocab_size)
    data_list = [sequence.pad_sequences(d, maxlen=overall_maxlen) for d in data_list]

    for d in data_list:
        np.random.shuffle(d)

    source_pos, source_neg, source_un_pos, source_un_neg, target_pos, target_neg, target_un_pos, target_un_neg = data_list

    # Each domain has a train set of size 1600, and a test set of size 400 with exactly balanced positive and negative examples
    # Only consider binary classification {pos: 1, neg: 0}
    source_train_y = np.concatenate((np.ones(800), np.zeros(800))).reshape(1600,1)
    source_test_y = np.concatenate((np.ones(200), np.zeros(200))).reshape(400, 1)
    target_train_y = np.concatenate((np.ones(800), np.zeros(800))).reshape(1600, 1)
    target_test_y = np.concatenate((np.ones(200), np.zeros(200))).reshape(400, 1)

    source_train_y = to_categorical(source_train_y, n_class)
    source_test_y = to_categorical(source_test_y, n_class)
    target_train_y = to_categorical(target_train_y, n_class)
    target_test_y = to_categorical(target_test_y, n_class)

    source_train_x = np.concatenate((source_pos[0:800], source_neg[0:800]))
    source_test_x = np.concatenate((source_pos[800:], source_neg[800:]))
    target_train_x = np.concatenate((target_pos[0:800], target_neg[0:800]))
    target_test_x = np.concatenate((target_pos[800:], target_neg[800:]))

    # Each domain has an additional unlabeled set of size 4000.
    source_un = np.concatenate((source_un_pos, source_un_neg))
    target_un = np.concatenate((target_un_pos, target_un_neg))

    # For each pair of source-target domain, the classifier is trained on the training set of the source domain and 
    # is evaluated on the test set of the target domain. The test set from source domain is used as development set.
    source_x, source_y = source_train_x, source_train_y
    dev_x, dev_y = source_test_x, source_test_y
    test_x, test_y = target_test_x, target_test_y
    source_un = np.concatenate((source_x, source_un))

    return vocab, overall_maxlen, source_x, source_y, dev_x, dev_y, test_x, test_y, source_un, target_un


