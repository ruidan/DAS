import argparse
import logging
import numpy as np
from time import time
import utils as U

logging.basicConfig(
                    # filename='out.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


##############################################################################################################################
# Parse arguments

parser = argparse.ArgumentParser()
# arguments related to datasets and data preprocessing
parser.add_argument("--dataset", dest="dataset", type=str, metavar='<str>', required=True, help="The name of the dataset (small_1|small_2|large|amazon)")
parser.add_argument("--source", dest="source", type=str, metavar='<str>', required=True, help="The name of the source domain")
parser.add_argument("--target", dest="target", type=str, metavar='<str>', required=True, help="The name of the source target")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=10000, help="Vocab size. '0' means no limit (default=0)")
parser.add_argument("--n-class", dest="n_class", type=int, metavar='<int>', default=3, help="The number of ouput classes")
parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='DAS', help="Model type (default=DAS)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")


# hyper-parameters related to network training
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")

# hyper-parameters related to network structure
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=300)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=300, help="CNN output dimension.(default=300)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
parser.add_argument("--cnn-activation", dest="cnn_activation", type=str, metavar='<str>', default='relu', help="The activation of CNN")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, input 0 (default=0.5)")
parser.add_argument("--discrepancy-obj", dest="minimize_discrepancy_obj", type=str, metavar='<str>', default='kl_loss', help="The loss for minimizing domain discrepancy (default=kl_loss)")

# hyper-parameters related to DAS objectives
parser.add_argument("--weight-discrepancy", dest="weight_discrepancy", type=float, metavar='<float>', default=200, help="The weight of the domain discrepancy minimization objective (lamda_1 in the paper)")
parser.add_argument("--weight-entropy", dest="weight_entropy", type=float, metavar='<float>', default=1.0, help="The weight of the target entropy objective (lamda_2 in the paper)")
parser.add_argument("--weight-uns", dest="weight_uns", type=float, metavar='<float>', default=3.0, help="The max value of the ensemble prediction objective weight (lamda_3 in the paper)")
parser.add_argument("--ensemble-prob", dest="ensemble_prob", type=float, metavar='<float>', default=0.5, help="The ensemble momentum (alpha in the paper)")

# random seed that affects data splits and parameter intializations
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")

args = parser.parse_args()
U.print_args(args)

# small_1 and small_2 denote eperimenal setting 1 and setting 2 on the small-scale dataset respectively.
# large denotes the large-scale dataset. Table 1(b) in the paper
# amazon denotes the amazon benchmark dataset (Blitzer et al., 2007). See appendix A in the paper.
assert args.dataset in {'small_1', 'small_2', 'large', 'amazon'}
assert args.model_type == 'DAS'

# The domains contained in each dataset
if args.dataset in {'small_1', 'small_2'}:
    assert args.source in {'book', 'electronics', 'beauty', 'music'}
    assert args.target in {'book', 'electronics', 'beauty', 'music'}
elif args.dataset == 'large':
    assert args.source in {'imdb', 'yelp2014', 'cell_phone', 'baby'}
    assert args.target in {'imdb', 'yelp2014', 'cell_phone', 'baby'}
else:
    # note that the book and electronics domains of amazon benchmark are different from those in small_1 and small_2
    assert args.source in {'book', 'dvd', 'electronics', 'kitchen'} 
    assert args.target in {'book', 'dvd', 'electronics', 'kitchen'}

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}

# In DAS, we use kl_loss for minimizing domain discrepancy. (See section 3.2 in paper)
assert args.minimize_discrepancy_obj in {'kl_loss', 'mmd'}

if args.seed > 0:
    np.random.seed(args.seed)


##############################################################################################################################
# Prepare data
if args.dataset == 'amazon':
    from read_amazon import get_data
else:
    from read import get_data

vocab, overall_maxlen, source_x, source_y, dev_x, dev_y, test_x, test_y, source_un, target_un = get_data(
    args.dataset, args.source, args.target, args.n_class, args.vocab_size)


print '------------ Traing Sets ------------'
print 'Number of labeled source examples: ', len(source_x)
print 'Number of total source examples (labeled+unlabeled): ', len(source_un)
print 'Number of unlabeled target examples: ', len(target_un)

print '------------ Development Set ------------'
print 'Size of development set: ', len(dev_x)

print '------------ Test Set -------------'
print 'Size of test set: ', len(test_x)




def batch_generator(data_list, batch_size):
    num = len(data_list[0])
    while True:
        excerpt = np.random.choice(num, batch_size)
        yield[data[excerpt] for data in data_list]

def batch_generator_large(data_list, batch_size):
    #######################################
    # Generate balanced labeled source examples.
    # Only used on large dataset as 
    # the training set is quite unbalanced.
    #######################################
    label_list = np.argmax(data_list[1], axis=-1)
    pos_inds = np.where(label_list==0)[0]
    neg_inds = np.where(label_list==1)[0]
    neu_inds = np.where(label_list==2)[0]

    while True:
        pos_sample = np.random.choice(pos_inds, batch_size/3)
        neg_sample = np.random.choice(neg_inds, batch_size/3)
        neu_sample = np.random.choice(neu_inds, batch_size/3+batch_size%3)
        excerpt = np.concatenate((pos_sample, neg_sample))
        excerpt = np.concatenate((excerpt, neu_sample))
        np.random.shuffle(excerpt)
        yield[data[excerpt] for data in data_list]



##############################################################################################################################
# Optimizer algorithm

from optimizers import get_optimizer
optimizer = get_optimizer(args)


###############################################################################################################################
# Building model

from models import create_model
import keras.backend as K

logger.info('  Building model')

def entropy(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_pred, y_pred), axis=-1)

def return_ypred(y_true, y_pred):
    return y_pred

model = create_model(args, overall_maxlen, vocab)
model.compile(optimizer=optimizer,
              loss={'source_probs': 'categorical_crossentropy', 'target_probs': entropy, 'discrepancy_loss': return_ypred, 'uns_loss': return_ypred},
              loss_weights={'source_probs': 1, 'target_probs': args.weight_entropy, 'discrepancy_loss': args.weight_discrepancy, 'uns_loss': 1},
              metrics={'source_probs': 'categorical_accuracy'})

###############################################################################################################################
# Training

from keras.utils.np_utils import to_categorical

# weight ramp-up function on the ensemble prediction objective
# w(t) in the paper.
def rampup(epoch):
    max_rampup_epochs = 30.0
    if epoch == 0:
        return 0

    elif epoch < args.epochs:
        p = min(max_rampup_epochs, float(epoch)) / max_rampup_epochs
        p = 1.0 - p
        return np.exp(-p*p*5.0)*args.weight_uns


from tqdm import tqdm
logger.info('----------------------------------------- Training Model ---------------------------------------------------------')

if args.dataset == 'large':
    source_gen = batch_generator_large([source_x, source_y], batch_size=args.batch_size)
else:
    source_gen = batch_generator([source_x, source_y], batch_size=args.batch_size)
source_un_gen = batch_generator([source_un], batch_size=args.batch_size)
target_un_gen = batch_generator([target_un], batch_size=args.batch_size)
overall_x = np.concatenate((source_un, target_un))
samples_per_epoch = len(overall_x)
batches_per_epoch = samples_per_epoch / args.batch_size
# Set the limit of batches_per_epoch to 500
batches_per_epoch = min(batches_per_epoch, 500)

#Initialize targets for unlabeled data. (See algorithm 1 in paper)
ensemble_prediction = np.zeros((len(overall_x), args.n_class)) 
targets = np.zeros((len(overall_x), args.n_class))
epoch_predictions = np.zeros((len(overall_x), args.n_class))


get_predictions = K.function([model.get_layer('uns_input').input, K.learning_phase()], [model.get_layer('uns_predictions').output])


best_valid_acc = 0
pred_probs = None

for ii in xrange(args.epochs):
    t0 = time()
    train_loss, source_loss, target_loss, dis_loss, uns_loss, train_metric = 0., 0., 0., 0., 0., 0.
    uns_gen = batch_generator([overall_x, targets], batch_size=args.batch_size)

    for b in tqdm(xrange(batches_per_epoch)):
        batch_source_x, batch_source_y = source_gen.next()
        batch_source_un = source_un_gen.next()[0]
        batch_target_un = target_un_gen.next()[0]
        batch_uns, batch_targets = uns_gen.next()

        train_loss_, source_loss_, target_loss_, dis_loss_, uns_loss_, train_metric_ = model.train_on_batch(
        [batch_source_x, batch_source_un, batch_target_un, batch_uns, batch_targets, np.full((args.batch_size, 1), rampup(ii))],
        {'source_probs': batch_source_y, 'target_probs': batch_source_y, 'discrepancy_loss': np.ones((args.batch_size, 1)) , 
        'uns_loss': np.ones((args.batch_size, 1))})

        train_loss += train_loss_ / batches_per_epoch
        source_loss += source_loss_ / batches_per_epoch
        target_loss += target_loss_ / batches_per_epoch
        uns_loss += uns_loss_ / batches_per_epoch
        dis_loss += dis_loss_ / batches_per_epoch
        train_metric += train_metric_ / batches_per_epoch

    # after the training of each epoch, compute predictions on unlabeled data 
    for ind in xrange(0, len(overall_x), args.batch_size):
        if ind+args.batch_size > len(overall_x):
            batch_inds = range(ind, len(overall_x))
        else:
            batch_inds = range(ind, ind+args.batch_size)
        batch_ = overall_x[batch_inds]
        batch_predictions = get_predictions([batch_, 0])[0]
        for i, j in enumerate(batch_inds):
            epoch_predictions[j] = batch_predictions[i]

    # compute ensemble predictions on unlabeled data
    ensemble_prediction = args.ensemble_prob*ensemble_prediction + (1-args.ensemble_prob)*epoch_predictions
    targets = ensemble_prediction / (1.0-args.ensemble_prob**(ii+1))
    targets = to_categorical(np.argmax(ensemble_prediction, axis=-1), args.n_class)  

    tr_time = time() - t0

    valid_loss, valid_source_loss, valid_target_loss, valid_dis_loss, valid_uns_loss, valid_metric = model.evaluate([dev_x, dev_x, dev_x, dev_x, dev_y, np.ones((len(dev_y), 1))],\
        {'source_probs': dev_y, 'target_probs': dev_y, 'discrepancy_loss': np.ones((len(dev_x),1)), 'uns_loss': np.ones((len(dev_x),1))}, batch_size=args.batch_size, verbose=1)
   

    logger.info('Epoch %d, train: %is' % (ii, tr_time))
    logger.info('[Train] loss: %.4f, [Source Classification] loss: %.4f, [Target Entropy] loss, %.4f, [Ensemble Prediction] loss: %.4f, [Discrepancy] loss: %.4f, metric: %.4f' \
                % (train_loss, source_loss, target_loss, uns_loss, dis_loss, train_metric))
    logger.info('[Validation] loss: %.4f, [Classification] loss: %.4f, [Entropy] loss, %.4f, [Ensemble Prediction] loss: %.4f, [Discrepancy] loss: %.4f, metric: %.4f' \
                % (valid_loss, valid_source_loss, valid_target_loss, valid_uns_loss, valid_dis_loss, valid_metric))
    

    if valid_metric > best_valid_acc:

        best_valid_acc = valid_metric
        print("------------- Best performance on dev set so far ==> evaluating on test set -------------")
        logger.info("------------- Best performance on dev set so far ==> evaluating on test set -------------\n")

        if args.dataset == 'large':
            #pad test set so that its size is dividible by batch_size
            append = args.batch_size-(len(test_y)%args.batch_size)
            test_x_ = np.concatenate((test_x, np.zeros((append, test_x.shape[1]))))
            test_y_ = np.concatenate((test_y, np.zeros((append, test_y.shape[1]))))

            pred_probs = model.predict([test_x_, test_x_, test_x_, test_x_, 
                test_y_, np.ones((len(test_y_), 1))], batch_size=args.batch_size, verbose=1)[0]

            pred_probs = pred_probs[:len(test_y)]
           

        else:
            pred_probs = model.predict([test_x, test_x, test_x, test_x, 
                test_y, np.ones((len(test_y), 1))], batch_size=args.batch_size, verbose=1)[0]

        
        from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
        preds = np.argmax(pred_probs, axis=-1)
        true = np.argmax(test_y, axis=-1)

        # Compute accuracy on test set
        logger.info("accuracy: "+ str(accuracy_score(true, preds)) + "\n")

        # Compute macro-f1 on test set
        p_macro, r_macro, f_macro, support_macro \
          = precision_recall_fscore_support(true, preds, average='macro')
        f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
        logger.info("macro-f1: "+str(f_macro) + "\n\n")


