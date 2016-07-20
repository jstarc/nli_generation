from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, RepeatVector, Flatten, Lambda
from keras.layers import Input, merge
from keras.models import Model
from keras import backend as K

from hierarchical_softmax import HierarchicalSoftmax
from hierarchical_softmax import hs_categorical_crossentropy
from common import make_fixed_embeddings
from attention import LstmAttentionLayer, FeedLSTM

import theano
import numpy as np

    
def gen_train(noise_examples, hidden_size, noise_dim, glove, hypo_len, version):
    if version == 9:
        return baseline_train(noise_examples, hidden_size, noise_dim, glove, 
                              hypo_len, version)        
    elif version == 6 or version == 7:
        return autoe_train(hidden_size, noise_dim, glove, hypo_len, version)

    prem_input = Input(shape=(None,), dtype='int32', name='prem_input')
    hypo_input = Input(shape=(hypo_len + 1,), dtype='int32', name='hypo_input')
    noise_input = Input(shape=(1,), dtype='int32', name='noise_input')
    train_input = Input(shape=(None,), dtype='int32', name='train_input')
    class_input = Input(shape=(3,), name='class_input')
    
    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, hypo_len + 1)(hypo_input)
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid', name='premise')(prem_embeddings)
    
    hypo_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid', name='hypo')(hypo_embeddings)
    noise_layer = Embedding(noise_examples, noise_dim, 
                            input_length = 1, name='noise_embeddings')(noise_input)
    flat_noise = Flatten(name='noise_flatten')(noise_layer)
    if version == 8:
        create_input = merge([class_input, flat_noise], mode='concat')
    if version == 5:
        create_input = flat_noise

    creative = Dense(hidden_size, name = 'cmerge')(create_input)
    attention = LstmAttentionLayer(output_dim=hidden_size, return_sequences=True, 
                    feed_state = True, name='attention') ([hypo_layer, premise_layer, creative])
               
    hs = HierarchicalSoftmax(len(glove), trainable = True, name='hs')([attention, train_input])
    
    inputs = [prem_input, hypo_input, noise_input, train_input, class_input]
    if version == 5:
        inputs = inputs[:4]    

    model_name = 'version' + str(version)
    model = Model(input=inputs, output=hs, name = model_name)
    model.compile(loss=hs_categorical_crossentropy, optimizer='adam')              
    
    return model

def baseline_train(noise_examples, hidden_size, noise_dim, glove, hypo_len, version):
    prem_input = Input(shape=(None,), dtype='int32', name='prem_input')
    hypo_input = Input(shape=(hypo_len + 1,), dtype='int32', name='hypo_input')
    noise_input = Input(shape=(1,), dtype='int32', name='noise_input')
    train_input = Input(shape=(None,), dtype='int32', name='train_input')
    class_input = Input(shape=(3,), name='class_input')
    concat_dim = hidden_size + noise_dim + 3
    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, hypo_len + 1)(hypo_input)

    premise_layer = LSTM(output_dim=hidden_size, return_sequences=False,
                            inner_activation='sigmoid', name='premise')(prem_embeddings)
    
    noise_layer = Embedding(noise_examples, noise_dim,
                            input_length = 1, name='noise_embeddings')(noise_input)
    flat_noise = Flatten(name='noise_flatten')(noise_layer)    
    merged = merge([premise_layer, class_input, flat_noise], mode='concat')
    creative = Dense(concat_dim, name = 'cmerge')(merged)
    fake_merge = Lambda(lambda x:x[0], output_shape=lambda x:x[0])([hypo_embeddings, creative])
    hypo_layer = FeedLSTM(output_dim=concat_dim, return_sequences=True,
                         feed_layer = creative, inner_activation='sigmoid', 
                         name='attention')([fake_merge])

    hs = HierarchicalSoftmax(len(glove), trainable = True, name='hs')([hypo_layer, train_input])
    inputs = [prem_input, hypo_input, noise_input, train_input, class_input]


    model_name = 'version' + str(version)
    model = Model(input=inputs, output=hs, name = model_name)
    model.compile(loss=hs_categorical_crossentropy, optimizer='adam')

    return model


def baseline_test(train_model, glove, batch_size):
    version = int(train_model.name[-1])
    hidden_size = train_model.get_layer('attention').output_shape[-1]    
    
    premise_input = Input(batch_shape=(batch_size, None, None))
    hypo_input = Input(batch_shape=(batch_size, 1), dtype='int32')
    creative_input = Input(batch_shape=(batch_size, None))
    train_input = Input(batch_shape=(batch_size, 1), dtype='int32')

    hypo_embeddings = make_fixed_embeddings(glove, 1)(hypo_input)
    hypo_layer = FeedLSTM(output_dim=hidden_size, return_sequences=True, 
                         stateful = True, trainable= False, feed_layer = premise_input,
                         name='attention')([hypo_embeddings])
    hs = HierarchicalSoftmax(len(glove), trainable = False, name ='hs')([hypo_layer, train_input])

    inputs = [hypo_input, creative_input, train_input]
    outputs = [hs]

    model = Model(input=inputs, output=outputs, name=train_model.name)
    model.compile(loss=hs_categorical_crossentropy, optimizer='adam')

    update_gen_weights(model, train_model)
    f_inputs = [train_model.get_layer('noise_embeddings').output,
                    train_model.get_layer('class_input').input,
                    train_model.get_layer('prem_input').input]
    func_noise = theano.function(f_inputs,  train_model.get_layer('cmerge').output,
                                     allow_input_downcast=True)

    return model, None, func_noise

def gen_test(train_model, glove, batch_size):
    
    version = int(train_model.name[-1])
    if version == 9:
        return baseline_test(train_model, glove, batch_size)
    hidden_size = train_model.get_layer('premise').output_shape[-1] 
    
    premise_input = Input(batch_shape=(batch_size, None, None))
    hypo_input = Input(batch_shape=(batch_size, 1), dtype='int32')
    creative_input = Input(batch_shape=(batch_size, None))
    train_input = Input(batch_shape=(batch_size, 1), dtype='int32')
    
    hypo_embeddings = make_fixed_embeddings(glove, 1)(hypo_input) 
    
    hypo_layer = LSTM(output_dim = hidden_size, return_sequences=True, stateful = True, unroll=False,
            trainable = False, inner_activation='sigmoid', name='hypo')(hypo_embeddings)
    
    att_inputs = [hypo_layer, premise_input] if version == 5 else [hypo_layer, premise_input, creative_input] 
    attention = LstmAttentionLayer(output_dim=hidden_size, return_sequences=True, stateful = True, unroll =False,
        trainable = False, feed_state = False, name='attention') \
            (att_inputs)

    hs = HierarchicalSoftmax(len(glove), trainable = False, name ='hs')([attention, train_input])
    
    inputs = [premise_input, hypo_input, creative_input, train_input]
    outputs = [hs]    
         
    model = Model(input=inputs, output=outputs, name=train_model.name)
    model.compile(loss=hs_categorical_crossentropy, optimizer='adam')
    
    update_gen_weights(model, train_model)
    
    func_premise = theano.function([train_model.get_layer('prem_input').input],
                                    train_model.get_layer('premise').output, 
                                    allow_input_downcast=True)
    if version == 5 or version == 8:   
        f_inputs = [train_model.get_layer('noise_embeddings').output]
        if version == 8:
            f_inputs += [train_model.get_layer('class_input').input]
       
        func_noise = theano.function(f_inputs, train_model.get_layer('cmerge').output, 
                                     allow_input_downcast=True)                            
    elif version == 6 or version == 7:
        noise_input = train_model.get_layer('reduction').output
        class_input = train_model.get_layer('class_input').input
        noise_output = train_model.get_layer('expansion').output
         
        func_noise = theano.function([noise_input, class_input], noise_output, 
                                      allow_input_downcast=True, on_unused_input='ignore') 
              
    return model, func_premise, func_noise

def update_gen_weights(test_model, train_model):
    version = int(train_model.name[-1])
    if version != 9:
        test_model.get_layer('hypo').set_weights(train_model.get_layer('hypo').get_weights())
    test_model.get_layer('attention').set_weights(train_model.get_layer('attention').get_weights())
    test_model.get_layer('hs').set_weights(train_model.get_layer('hs').get_weights()) 
    
def word_loss(y_true, y_pred):
    return K.mean(hs_categorical_crossentropy(y_true, y_pred))
def cc_loss(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_pred, y_true))


def autoe_train(hidden_size, noise_dim, glove, hypo_len, version):

    prem_input = Input(shape=(None,), dtype='int32', name='prem_input')
    hypo_input = Input(shape=(hypo_len + 1,), dtype='int32', name='hypo_input')
    train_input = Input(shape=(None,), dtype='int32', name='train_input')
    class_input = Input(shape=(3,), name='class_input')

    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, hypo_len + 1)(hypo_input)
    premise_encoder = LSTM(output_dim=hidden_size, return_sequences=True,
                            inner_activation='sigmoid', name='premise_encoder')(prem_embeddings)

    hypo_encoder = LSTM(output_dim=hidden_size, return_sequences=True,
                            inner_activation='sigmoid', name='hypo_encoder')(hypo_embeddings)
    class_encoder = Dense(hidden_size, activation='tanh')(class_input)

    encoder = LstmAttentionLayer(output_dim=hidden_size, return_sequences=False,
                  feed_state = True, name='encoder') ([hypo_encoder, premise_encoder, class_encoder])
    if version == 6:
        reduction = Dense(noise_dim, name='reduction', activation='tanh')(encoder)
    elif version == 7:
        z_mean = Dense(noise_dim, name='z_mean')(encoder)
        z_log_sigma = Dense(noise_dim, name='z_log_sigma')(encoder)
          
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(64, noise_dim,),
                              mean=0., std=0.01)
            return z_mean + K.exp(z_log_sigma) * epsilon
        reduction = Lambda(sampling, output_shape=lambda sh: (sh[0][0], noise_dim,), name = 'reduction')([z_mean, z_log_sigma])
        def vae_loss(args):
            z_mean, z_log_sigma = args
            return - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)    
        vae = Lambda(vae_loss, output_shape=lambda sh: (sh[0][0], 1,), name = 'vae_output')([z_mean, z_log_sigma])

    merged = merge([class_input, reduction], mode='concat')
    creative = Dense(hidden_size, name = 'expansion', activation ='tanh')(merged)
    premise_decoder = LSTM(output_dim=hidden_size, return_sequences=True,
                            inner_activation='sigmoid', name='premise')(prem_embeddings)
    
    hypo_decoder = LSTM(output_dim=hidden_size, return_sequences=True,
                            inner_activation='sigmoid', name='hypo')(hypo_embeddings)
    attention = LstmAttentionLayer(output_dim=hidden_size, return_sequences=True,
                     feed_state = True, name='attention') ([hypo_decoder, premise_decoder, creative])

    hs = HierarchicalSoftmax(len(glove), trainable = True, name='hs')([attention, train_input])

    inputs = [prem_input, hypo_input, train_input, class_input]

    model_name = 'version' + str(version)
    model = Model(input=inputs, output=(hs if version == 6 else [hs, vae]), name = model_name)
    if version == 6:
        model.compile(loss=hs_categorical_crossentropy, optimizer='adam')
    elif version == 7:
        def minimize(y_true, y_pred):
            return y_pred
        def metric(y_true, y_pred):
            return K.mean(y_pred)
        model.compile(loss=[hs_categorical_crossentropy, minimize], metrics={'hs':word_loss, 'vae_output': metric}, optimizer='adam')
    return model


def count_params(model, explain = False):
    params = {}
    params['total'] = model.count_params()
    params['non_trainable'] = 0
    params['trainable'] = 0
    params['noise_embeddings'] = 0
    params['hs'] = 0
    for layer in model.layers:
        l_params = layer.count_params()
        if l_params > 0:
            if layer.trainable:
                if layer.name == 'hs':
                    params['hs'] += l_params
                elif layer.name == 'noise_embeddings':
                    params['noise_embeddings'] += l_params
                else:
                    params['trainable'] += l_params
            else:
                params['non_trainable'] += l_params
            if explain:
                print layer.name, l_params

    for k in params:
        params[k] = int(round(params[k] / 1000.0))
    return params
