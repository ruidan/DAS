import numpy as np
import logging
import codecs
from keras.layers import Dense, Dropout, Activation, Embedding, Input
from keras.models import Model
import keras.backend as K
from my_layers import Conv1DWithMasking, Max_over_time, KL_loss, Ensemble_pred_loss, mmd_loss
from keras.constraints import maxnorm


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)




def create_model(args, overal_maxlen, vocab):
        
   
    ##############################################################################################################################
    # Custom CNN kernel initializer
    # Use the initialization from Kim et al. (2014) for CNN kernel initialization. 
    def my_init(shape, dtype=K.floatx()):
        return 0.01 * np.random.standard_normal(size=shape)



    ##############################################################################################################################
    # Funtion that loads word embeddings from Glove vectors
    def init_emb(emb_matrix, vocab, emb_file):
        print 'Loading word embeddings ...'
        counter = 0.
        pretrained_emb = open(emb_file)
        for line in pretrained_emb:
            tokens = line.split()
            if len(tokens) != 301:
                continue
            word = tokens[0]
            vec = tokens[1:]
            try:
                emb_matrix[0][vocab[word]] = vec
                counter += 1
            except KeyError:
                pass
               
        pretrained_emb.close()
        logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))
            
        return emb_matrix


    ##############################################################################################################################
    # Create Model
    
    cnn_padding='same'   
    vocab_size = len(vocab)

    
    if args.model_type == 'DAS':
        print '\n'
        logger.info('Building model for DAS')

        # labeled source examples
        source_input = Input(shape=(overal_maxlen,), dtype='int32', name='source_input')
        # unlabeled source examples (this includes all source examples with and without labels)
        source_un_input = Input(shape=(overal_maxlen,), dtype='int32', name='source_un_input')
        # unlabeled target examples
        target_un_input = Input(shape=(overal_maxlen,), dtype='int32', name='target_un_input')
        # all examples from both source and target domains 
        uns_input = Input(shape=(overal_maxlen,), dtype='int32', name='uns_input')
        # estimated sentiment labels for all examples
        uns_target = Input(shape=(args.n_class,), dtype=K.floatx(), name='uns_target')
        # ramp-up weight
        uns_weight = Input(shape=(1, ), dtype=K.floatx(), name='uns_weight')

        word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')
        source_output = word_emb(source_input)
        source_un_output = word_emb(source_un_input)
        target_un_output = word_emb(target_un_input)
        uns_output = word_emb(uns_input)


        print 'use a cnn layer'
        conv = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=args.cnn_window_size, \
              activation=args.cnn_activation, padding=cnn_padding, kernel_initializer=my_init, name='cnn')
        source_output = conv(source_output)
        source_un_output = conv(source_un_output)
        target_un_output = conv(target_un_output)
        uns_output = conv(uns_output)
        
        print 'use max_over_time as aggregation function'
        source_output = Max_over_time(name='mot')(source_output)
        source_un_output = Max_over_time()(source_un_output)
        target_un_output = Max_over_time()(target_un_output)
        uns_output = Max_over_time()(uns_output)

        if args.minimize_discrepancy_obj == 'kl_loss':
            dis_loss = KL_loss(args.batch_size, name='discrepancy_loss')([source_un_output, target_un_output])
        elif args.minimize_discrepancy_obj == 'mmd':
            dis_loss = mmd_loss(args.batch_size, name='discrepancy_loss')([source_un_output, target_un_output])
        else:
            raise NotImplementedError


        if args.weight_discrepancy > 0:
            print 'Minimize domain discrepancy between source and target via %s'%(args.minimize_discrepancy_obj)

        if args.dropout_prob > 0:
            print 'use dropout layer'
            source_output = Dropout(args.dropout_prob)(source_output)
            target_un_output = Dropout(args.dropout_prob)(target_un_output)
            uns_output = Dropout(args.dropout_prob)(uns_output)
            

        clf = Dense(args.n_class, kernel_constraint=maxnorm(3), name='dense')
        source_output = clf(source_output)
        target_output = clf(target_un_output)
        uns_output = clf(uns_output)
        source_probs = Activation('softmax', name='source_probs')(source_output)
        target_probs = Activation('softmax', name='target_probs')(target_output)
        uns_probs = Activation('softmax', name='uns_predictions')(uns_output)

        uns_pred_loss = Ensemble_pred_loss(name='uns_loss')([uns_probs, uns_target, uns_weight])
        if args.weight_uns > 0:
            print 'Use ensemble prediction on unlabeled data for semi-supervised training'


        model = Model(inputs=[source_input, source_un_input, target_un_input, uns_input, uns_target, uns_weight], 
            outputs=[source_probs, target_probs, dis_loss, uns_pred_loss])

    else:
        raise NotImplementedError
    
    
    logger.info('  Done')
    print '\n'

    
    

    ##############################################################################################################################
    # Initialize embeddings if embedding path is given

    if args.emb_path:
        # It takes around 3 mininutes to load pre-trained word embeddings.
        model.get_layer('word_emb').set_weights(init_emb(model.get_layer('word_emb').get_weights(), vocab, args.emb_path))

    return model
    
