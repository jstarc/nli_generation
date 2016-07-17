from keras.layers.recurrent import LSTM
from keras.layers.core import Lambda, Dense
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from common import make_fixed_embeddings


def discriminator(glove, hidden_size):
    
    hypo_input = Input(shape=(None,), dtype='int32')
    embeds = make_fixed_embeddings(glove, None)(hypo_input)
    lstm = LSTM(hidden_size, inner_activation='sigmoid')(embeds)
    output = Dense(1, activation='sigmoid')(lstm)
    discriminator = Model([hypo_input], output)
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

def adverse_model(discriminator):
    
    train_input = Input(shape=(None,), dtype='int32')
    hypo_input = Input(shape=(None,), dtype='int32')
    
    def margin_opt(inputs):
        assert len(inputs) == 2, ('Margin Output needs '
                              '2 inputs, %d given' % len(inputs))
        return K.log(inputs[0]) + K.log(1-inputs[1])
            
    margin = Lambda(margin_opt, output_shape=(lambda s : (None, 1)))\
               ([discriminator(train_input), discriminator(hypo_input)])
    adverserial = Model([train_input, hypo_input], margin)
    
    adverserial.compile(loss=minimize, optimizer='adam')
    return adverserial

def minimize(y_true, y_pred):
        return K.abs(K.mean(y_pred, axis=-1))

def reinit(ad_model):
    ad_model.compile(loss=minimize, optimizer='adam')
    
    
    
