from keras.layers.recurrent import LSTM, time_distributed_dense
from keras.backend import theano_backend as K
from keras.engine import InputSpec


class FeedLSTM(LSTM):

    def __init__(self, feed_layer = None , **kwargs):
        self.feed_layer = feed_layer  
        self.supports_masking = False
    
        super(FeedLSTM, self).__init__(**kwargs)

    def set_state(self, noise):
        K.set_value(self.states[1], noise)     
            
    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        if self.feed_layer is not None:
            initial_states[1] =  self.feed_layer
        return initial_states
               

class LstmAttentionLayer(LSTM):

    def __init__(self, feed_state = False, **kwargs):
        self.feed_state = feed_state
        self.supports_masking = False
        super(LstmAttentionLayer, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0][0], input_shape[0][1], self.output_dim)
        else:
            return (input_shape[0][0], self.output_dim)
            
    def compute_mask(self, input, mask):
        return None
        
    def call(self, x, mask=None):
        return super(LSTM, self).call(x, None)
    
    def build(self, input_shape):
        
        self.input_spec = [InputSpec(shape=shape) for shape in input_shape]
        
        input_dim = input_shape[1][2]
        self.input_dim = input_dim
        
        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None]

        self.W_s = self.init((self.output_dim, self.output_dim))
        self.W_t = self.init((self.output_dim, self.output_dim))
        self.W_a = self.init((self.output_dim, self.output_dim))
        self.w_e = K.zeros((self.output_dim,))

        self.W_i = self.init((2 * self.output_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((2 * self.output_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim,))

        self.W_c = self.init((2 * self.output_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((2 * self.output_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))
        
        
        self.trainable_weights = [self.W_s, self.W_t, self.W_a, self.w_e,
                       self.W_i, self.U_i, self.b_i,
                       self.W_c, self.U_c, self.b_c,
                       self.W_f, self.U_f, self.b_f,
                       self.W_o, self.U_o, self.b_o]
    def preprocess_input(self, x):
        return x[0]    
 
    def set_state(self, noise):
        K.set_value(self.states[1], noise)
        
    def get_constants(self, x):
        return [x[1], K.dot(x[1], self.W_s)]
        
    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x[0])  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.output_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        if self.feed_state:
            initial_states[1] =  x[2]

        return initial_states

    def step(self, x, states):
        h_s = states[2]        
        P_j = states[3] 
        
        P_t = K.dot(x, self.W_t)
        P_a = K.dot(states[0], self.W_a)
        sum3 = P_j + P_t.dimshuffle((0,'x',1)) + P_a.dimshuffle((0,'x',1))
        E_kj = K.tanh(sum3).dot(self.w_e)
        Alpha_kj = K.softmax(E_kj)
        weighted = h_s * Alpha_kj.dimshuffle((0,1,'x'))
        a_k = weighted.sum(axis = 1)
        m_k = K.T.concatenate([a_k, x], axis = 1)
        
        x_i = K.dot(m_k, self.W_i) + self.b_i
        x_f = K.dot(m_k, self.W_f) + self.b_f
        x_c = K.dot(m_k, self.W_c) + self.b_c
        x_o = K.dot(m_k, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(states[0], self.U_i))
        f = self.inner_activation(x_f + K.dot(states[0], self.U_f))
        c = f * states[1] + i * self.activation(x_c + K.dot(states[0], self.U_c))
        o = self.inner_activation(x_o + K.dot(states[0], self.U_o))
        h = o * self.activation(c)
        
        return h, [h, c]

    def get_config(self):
        config = {'feed_state':self.feed_state}
        base_config = super(LstmAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        
        
        
        
