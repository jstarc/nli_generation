from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from attention import LstmAttentionLayer
from common import make_fixed_embeddings

def attention_model(hidden_size, glove):
        
    prem_input = Input(shape=(None,), dtype='int32')
    hypo_input = Input(shape=(None,), dtype='int32')
    
    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, None)(hypo_input)
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid')(prem_embeddings)
    hypo_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid')(hypo_embeddings)    
    attention = LstmAttentionLayer(output_dim = hidden_size) ([hypo_layer, premise_layer])
    final_dense = Dense(3, activation='softmax')(attention)
    
    model = Model(input=[prem_input, hypo_input], output=final_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model  


def attention_bnorm_model(hidden_size, glove):

    prem_input = Input(shape=(None,), dtype='int32')
    hypo_input = Input(shape=(None,), dtype='int32')

    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, None)(hypo_input)
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True,
                            inner_activation='sigmoid')(prem_embeddings)
    premise_bn = BatchNormalization()(premise_layer)
    hypo_layer = LSTM(output_dim=hidden_size, return_sequences=True,
                            inner_activation='sigmoid')(hypo_embeddings)
    hypo_bn = BatchNormalization()(hypo_layer)
    attention = LstmAttentionLayer(output_dim = hidden_size) ([hypo_bn, premise_bn])
    att_bn = BatchNormalization()(attention)
    final_dense = Dense(3, activation='softmax')(att_bn)

    model = Model(input=[prem_input, hypo_input], output=final_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
