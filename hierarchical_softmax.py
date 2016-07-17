from keras import backend as K
from keras import initializations
from keras.backend.common import _EPSILON 	
from keras.engine.topology import Layer
from keras.engine import InputSpec

from theano.tensor.nnet import h_softmax
import theano.tensor as T



class HierarchicalSoftmax(Layer): 

    def __init__(self, output_dim, init='glorot_uniform', **kwargs):
        self.init = initializations.get(init)
        self.output_dim = output_dim
        
        def hshape(n):
            from math import sqrt, ceil
            l1 = ceil(sqrt(n))
            l2 = ceil(n / l1)
            return int(l1), int(l2)
      
        self.n_classes, self.n_outputs_per_class = hshape(output_dim)
        super(HierarchicalSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.input_spec = [InputSpec(shape=shape) for shape in input_shape]
        input_dim =  self.input_spec[0].shape[-1]
        self.W1 = self.init((input_dim, self.n_classes), name='{}_W1'.format(self.name))
        self.b1 = K.zeros((self.n_classes,),  name='{}_b1'.format(self.name))
        self.W2 = self.init((self.n_classes, input_dim, self.n_outputs_per_class), name='{}_W2'.format(self.name))
        self.b2 = K.zeros((self.n_classes, self.n_outputs_per_class),  name='{}_b2'.format(self.name))

        self.trainable_weights = [self.W1, self.b1, self.W2, self.b2]


    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], None)

    def call(self, X, mask=None):
       
        input_shape = self.input_spec[0].shape
        
        x = K.reshape(X[0], (-1, input_shape[2]))
        target = X[1].flatten() if self.trainable else None
        
        Y = h_softmax(x, K.shape(x)[0], self.output_dim, 
                              self.n_classes, self.n_outputs_per_class,
                              self.W1, self.b1, self.W2, self.b2, target)
        
        output_dim = 1 if self.trainable else self.output_dim    
        input_length = K.shape(X[0])[1]
       
        y = K.reshape(Y, (-1, input_length, output_dim))
        return y
        
    

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__}
        base_config = super(HierarchicalSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
def hs_categorical_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)    
    return T.nnet.categorical_crossentropy(y_pred, y_true)


    
