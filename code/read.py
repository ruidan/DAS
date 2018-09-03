import codecs
import operator
import numpy as np
import re
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def create_vocab(file_list, vocab_size, skip_len):

    print 'Creating vocab ...'

    total_words, unique_words = 0, 0
    word_freqs = {}

    for f in file_list:
        fin = codecs.open(f, 'r', 'utf-8')
        for line in fin:
            words = line.split()
            if skip_len > 0 and len(words) > skip_len:
                continue

            for w in words:
                if not bool(num_regex.match(w)):
                    try:
                        word_freqs[w] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[w] = 1
                    total_words += 1
        fin.close()

    print ('  %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    print (' keep the top %i words' % vocab_size)

    return vocab


def create_data(vocab, text_path, label_path, domain, n_class, skip_top, skip_len, replace_non_vocab):
    data = []
    label = [] # {pos: 0, neg: 1, neu: 2}
    f = codecs.open(text_path, 'r', 'utf-8')
    f_l = codecs.open(label_path, 'r', 'utf-8')
    num_hit, unk_hit, skip_top_hit, total = 0., 0., 0., 0.
    pos_count, neg_count, neu_count = 0, 0, 0
    max_len = 0

    for line, score in zip(f, f_l):
        word_indices = []
        words = line.split()
        if skip_len > 0 and len(words) > skip_len:
            continue

        score = float(score.strip())
        if domain == 'imdb':
            if score < 5:
                neg_count += 1
                label.append(1)
            elif score > 6:
                pos_count += 1
                label.append(0)
            else:
                if n_class == 3:
                    neu_count += 1
                    label.append(2)
                else:
                    continue
    
        elif domain in {'yelp2014', 'book', 'electronics', 'beauty', 'music', 'cell_phone', 'baby'}:
            if score < 3:
                neg_count += 1
                label.append(1)
            elif score > 3:
                pos_count += 1
                label.append(0)
            else:
                if n_class == 3:
                    neu_count += 1
                    label.append(2)
                else:
                    continue
        else:
            print 'No such domain!'
            break


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

    f.close()
    f_l.close()

    print('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%, <skip_top> hit rate: %.2f%%' \
                        % (100*num_hit/total, 100*unk_hit/total, 100*skip_top_hit/total))


    print domain
    print 'pos count: ', pos_count
    print 'neg count: ', neg_count
    print 'neu count: ', neu_count
    return np.array(data), np.array(label), max_len



def prepare_data(dataset, source_domain, target_domain, n_class, vocab_size=0, skip_len=0, skip_top=0, replace_non_vocab=1):

    if dataset == 'small_1':
        text_list = ['../data/small/%s/set1_text.txt'%source_domain, 
                     '../data/small/%s/set1_text.txt'%target_domain]

        score_list = ['../data/small/%s/set1_label.txt'%source_domain, 
                      '../data/small/%s/set1_label.txt'%target_domain]

        domain_list = [source_domain, target_domain]

    elif dataset == 'small_2':
        text_list = ['../data/small/%s/set1_text.txt'%source_domain, 
                     '../data/small/%s/set1_text.txt'%target_domain,
                     '../data/small/%s/set2_text.txt'%source_domain, 
                     '../data/small/%s/set2_text.txt'%target_domain]

        score_list = ['../data/small/%s/set1_label.txt'%source_domain, 
                      '../data/small/%s/set1_label.txt'%target_domain,
                      '../data/small/%s/set2_label.txt'%source_domain, 
                      '../data/small/%s/set2_label.txt'%target_domain]

        domain_list = [source_domain, target_domain, source_domain, target_domain]

    else:
        text_list = ['../data/large/%s/text.txt'%source_domain, 
                     '../data/large/%s/text.txt'%target_domain]

        score_list = ['../data/large/%s/label.txt'%source_domain, 
                      '../data/large/%s/label.txt'%target_domain]

        domain_list = [source_domain, target_domain]


    vocab = create_vocab(text_list, vocab_size, skip_len)

    

    data_list = []
    label_list = []
    overall_max_len = 0
    for f, f_l, domain in zip(text_list, score_list, domain_list):
        data, label, max_len = create_data(vocab, f, f_l, domain, n_class, skip_top, skip_len, replace_non_vocab)
        data_list.append(data)
        label_list.append(label)
        if max_len > overall_max_len:
            overall_max_len = max_len

    return vocab, data_list, label_list, overall_max_len


def get_data(dataset, source_domain, target_domain, n_class, vocab_size=0):
    assert dataset in ['small_1', 'small_2', 'large']

    vocab, data_list, label_list, overall_maxlen = prepare_data(dataset, source_domain, target_domain, n_class, vocab_size)
    
    data_list = [sequence.pad_sequences(d, maxlen=overall_maxlen) for d in data_list]
    label_list = [to_categorical(l, n_class) for l in label_list]

    if dataset == 'large':
        # when using the large-scale dataset, we need to sample 1k balanced dev set from labeled source data
        labels = np.argmax(label_list[0], axis=-1)
        pos_inds = np.where(labels==0)[0]
        neg_inds = np.where(labels==1)[0]
        neu_inds = np.where(labels==2)[0]
        np.random.shuffle(pos_inds)
        np.random.shuffle(neg_inds)
        np.random.shuffle(neu_inds)

        dev_inds = np.concatenate((pos_inds[:333], neg_inds[:333]))
        dev_inds = np.concatenate((dev_inds, neu_inds[:334]))
        train_inds = np.concatenate((pos_inds[333:], neg_inds[333:]))
        train_inds = np.concatenate((train_inds, neu_inds[334:]))

        source_x, source_y = data_list[0][train_inds], label_list[0][train_inds]
        dev_x, dev_y = data_list[0][dev_inds], label_list[0][dev_inds]

    else:
        #On small-scale dataset, randomly select 1k examples from set1 of source domain as dev set
        inds = np.random.permutation(data_list[0].shape[0])
        dev_inds, train_inds = inds[:1000], inds[1000:]
        source_x, source_y = data_list[0][train_inds], label_list[0][train_inds]
        dev_x, dev_y = data_list[0][dev_inds], label_list[0][dev_inds]

    test_x, test_y = data_list[1], label_list[1]

    if dataset in ['small_1', 'large']:
        source_un = data_list[0]
        target_un = data_list[1]

    else:
        source_un = np.concatenate((data_list[0], data_list[2]))
        target_un = data_list[3]

    return vocab, overall_maxlen, source_x, source_y, dev_x, dev_y, test_x, test_y, source_un, target_un










