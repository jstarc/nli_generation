import os
import load_data
import numpy as np

from keras.backend import theano_backend as K 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.generic_utils import Progbar
from keras.callbacks import Callback
import generative_models as gm
from common import CsvHistory
from common import merge_result_batches
import adverse_models as am
from collections import Counter
from scipy.stats import entropy

def train(train, dev, model, model_dir, batch_size, glove, beam_size,
          samples_per_epoch, val_samples, cmodel, epochs):
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)

    hypo_len = model.get_layer('hypo_input').input_shape[1] -1
    ne = model.get_layer('noise_embeddings')
    vae = model.get_layer('vae_output')
    
    g_train = train_generator(train, batch_size, hypo_len, 
                               'class_input' in model.input_names, ne, vae)
    saver = ModelCheckpoint(model_dir + '/weights.hdf5', monitor = 'hypo_loss', mode = 'min', save_best_only = True)
    #saver = ModelCheckpoint(model_dir + '/weights{epoch:02d}.hdf5')
    #es = EarlyStopping(patience = 4,  monitor = 'hypo_loss', mode = 'min')
    csv = CsvHistory(model_dir + '/history.csv')

     
    
    gtest = gm.gen_test(model, glove, batch_size)
    noise_size = ne.output_shape[-1] if ne else model.get_layer('expansion').input_shape[-1] 
    cb = ValidateGen(dev, gtest, beam_size, hypo_len, val_samples, noise_size, glove, cmodel, True, True)
    
    hist = model.fit_generator(g_train, samples_per_epoch = samples_per_epoch, nb_epoch = epochs,  
                              callbacks = [cb, saver, csv])
    return hist
            

def train_generator(train, batch_size, hypo_len, cinput, ninput, vae):
    while True:
         mb = load_data.get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
        
         for i, train_index in mb:
             if len(train_index) != batch_size:
                 continue
             padded_p = train[0][train_index]
             padded_h = train[1][train_index]
             label = train[2][train_index]
             hypo_input = np.concatenate([np.zeros((batch_size, 1)), padded_h], axis = 1)
             train_input = np.concatenate([padded_h, np.zeros((batch_size, 1))], axis = 1)
             inputs = [padded_p, hypo_input] + ([train_index[:, None]] if ninput else []) + [train_input]
             if cinput:
                 inputs.append(label)
             outputs = [np.ones((batch_size, hypo_len + 1, 1))]
             if vae:
                 outputs += [np.zeros(batch_size)]
             yield (inputs, outputs)

                    
def generative_predict_beam(test_model, premises, noise_batch, class_indices, return_best, hypo_len):
    
    core_model, premise_func, noise_func = test_model
    version = int(core_model.name[-1])

    batch_size = core_model.input_layers[0].input_shape[0]
    
    beam_size = batch_size / len(premises)
    dup_premises = np.repeat(premises, beam_size, axis = 0)
    premise = premise_func(dup_premises) if version != 9 else None 
      
    class_input = np.repeat(class_indices, beam_size, axis = 0)
    embed_vec = np.repeat(noise_batch, beam_size, axis = 0)
    if version == 8:
        noise = noise_func(embed_vec, class_input)
    elif version == 6 or version == 7:
        noise = noise_func(embed_vec[:,-1,:], class_input)
    elif version == 9:
        noise = noise_func(embed_vec, class_input, dup_premises) 
    elif version == 5:
        noise = noise_func(embed_vec)
 
    core_model.reset_states()
    core_model.get_layer('attention').set_state(noise)

    word_input = np.zeros((batch_size, 1))
    result_probs = np.zeros(batch_size)
    debug_probs = np.zeros((hypo_len, batch_size))
    lengths = np.zeros(batch_size)
    words = None
    probs = None
   
    for i in range(hypo_len):
        data = [premise, word_input, noise, np.zeros((batch_size,1))]
        if version == 9:
            data = data[1:]
        preds = core_model.predict_on_batch(data)
        preds = np.log(preds)
        split_preds = np.array(np.split(preds, len(premises)))
        if probs is None:
            if beam_size == 1:
                word_input =  np.argmax(split_preds[:, 0, 0], axis = 1)[:,None]
            else:
                word_input = np.argpartition(-split_preds[:, 0, 0], beam_size)[:,:beam_size]
            probs = split_preds[:,0,0][np.arange(len(premises))[:, np.newaxis],[word_input]].ravel()
            word_input= word_input.ravel()[:,None]
            words = np.array(word_input)
            debug_probs[0] = probs 
        else:
            split_cprobs =  (preds[:,-1,:] + probs[:, None]).reshape((len(premises), -1))
            if beam_size == 1:
                max_indices = np.argmax(split_cprobs, axis = 1)[:,None]
            else:
                max_indices = np.argpartition(-split_cprobs, beam_size)[:,:beam_size]
            probs = split_cprobs[np.arange(len(premises))[:, np.newaxis],[max_indices]].ravel()
            word_input = (max_indices % preds.shape[-1]).ravel()[:,None]
            state_indices = (max_indices / preds.shape[-1]) + np.arange(0, batch_size, beam_size)[:, None]
            state_indices = state_indices.ravel()
            shuffle_states(core_model, state_indices)
            words = np.concatenate([words[state_indices], word_input], axis = -1)
            debug_probs = debug_probs[:, state_indices]
            debug_probs[i] = probs - np.sum(debug_probs, axis = 0)
        lengths += 1 * (word_input[:,0] > 0).astype('int')
        if (np.sum(word_input) == 0):
            words = np.concatenate([words, np.zeros((batch_size, hypo_len - words.shape[1]))], 
                                    axis = -1)
            break
    result_probs = probs / -lengths   
    if return_best:
        best_ind = np.argmin(np.array(np.split(result_probs, len(premises))), axis =1) + np.arange(0, batch_size, beam_size)
        return words[best_ind], result_probs[best_ind]
    else:
        return words, result_probs, debug_probs
    
def shuffle_states(graph_model, indices):
    for l in graph_model.layers:
        if getattr(l, 'stateful', False): 
            for s in l.states:
                K.set_value(s, s.get_value()[indices])
                
                
def val_generator(dev, gen_test, beam_size, hypo_len, noise_size):
    batch_size = gen_test[0].input_layers[0].input_shape[0]
    
    per_batch  = batch_size / beam_size
    while True:
        mb = load_data.get_minibatches_idx(len(dev[0]), per_batch, shuffle=False)
        for i, train_index in mb:
            if len(train_index) != per_batch:
               continue
            premises = dev[0][train_index]
            noise_input = np.random.normal(scale=0.11, size=(per_batch, 1, noise_size))
            class_indices = dev[2][train_index] 
            words, loss = generative_predict_beam(gen_test, premises, noise_input,
                             class_indices, True, hypo_len)
            yield premises, words, loss, noise_input, class_indices

def single_generate(premise, label, gen_test, beam_size, hypo_len, noise_size, noise_input = None):
    batch_size = gen_test[0].input_layers[0].input_shape[0]
    per_batch  = batch_size / beam_size
    premises = [premise] * per_batch
    if noise_input is None:
        noise_input = np.random.normal(scale=0.11, size=(per_batch, 1, noise_size))
    class_indices = np.ones(per_batch) * label
    class_indices = load_data.convert_to_one_hot(class_indices, 3)
    words, loss = generative_predict_beam(gen_test, premises, noise_input,
                             class_indices, True, hypo_len)

    return words

def validate(dev, gen_test, beam_size, hypo_len, samples, noise_size, glove, cmodel = None, adverse = False, 
                 diverse = False):
    vgen = val_generator(dev, gen_test, beam_size, hypo_len, noise_size)
    p = Progbar(samples)
    batchez = []
    while p.seen_so_far < samples:
        batch = next(vgen)
        preplexity = np.mean(np.power(2, batch[2]))
        loss = np.mean(batch[2])
        losses = [('hypo_loss',loss),('perplexity', preplexity)]
        if cmodel is not None:
            ceval = cmodel.evaluate([batch[0], batch[1]], batch[4], verbose = 0)
            losses += [('class_loss', ceval[0]), ('class_acc', ceval[1])]
            probs = cmodel.predict([batch[0], batch[1]], verbose = 0)
            losses += [('class_entropy', np.mean(-np.sum(probs * np.log(probs), axis=1)))]
        
        p.add(len(batch[0]), losses)
        batchez.append(batch)
    batchez = merge_result_batches(batchez)
    
    res = {}
    if adverse:
        val_loss = adverse_validation(dev, batchez, glove)
        print 'adverse_loss:', val_loss
        res['adverse_loss'] = val_loss
    if diverse:
        div, _, _, _ = diversity(dev, gen_test, beam_size, hypo_len, noise_size, 64, 32)
        res['diversity'] = div
    print
    for val in p.unique_values:
        arr = p.sum_values[val]
        res[val] = arr[0] / arr[1]
    return res

def adverse_validation(dev, batchez, glove):
    samples = len(batchez[1])
    discriminator = am.discriminator(glove, 50)
    ad_model = am.adverse_model(discriminator)
    res = ad_model.fit([dev[1][:samples], batchez[1]], np.zeros(samples), validation_split=0.1, 
                       verbose = 0, nb_epoch = 20, callbacks = [EarlyStopping(patience=2)])
    return np.min(res.history['val_loss'])

def diversity(dev, gen_test, beam_size, hypo_len, noise_size, per_premise, samples):
    step = len(dev[0]) / samples
    sind = [i * step for i in range(samples)]
    p = Progbar(per_premise * samples)
    for i in sind:
        hypos = []
        unique_words = []
        hypo_list = []
        premise = dev[0][i]
        prem_list = set(cut_zeros(list(premise)))        
        while len(hypos) < per_premise:
            label = np.argmax(dev[2][i])
            words = single_generate(premise, label, gen_test, beam_size, hypo_len, noise_size)
            hypos += [str(ex) for ex in words]
            unique_words += [int(w) for ex in words for w in ex if w > 0]
            hypo_list += [set(cut_zeros(list(ex))) for ex in words]
        
        jacks = []  
        prem_jacks = []
        for u in range(len(hypo_list)):
            sim_prem = len(hypo_list[u] & prem_list)/float(len(hypo_list[u] | prem_list))
            prem_jacks.append(sim_prem)
            for v in range(u+1, len(hypo_list)):
                sim = len(hypo_list[u] & hypo_list[v])/float(len(hypo_list[u] | hypo_list[v]))
                jacks.append(sim)
        avg_dist_hypo = 1 -  np.mean(jacks)
        avg_dist_prem = 1 -  np.mean(prem_jacks)
        d = entropy(Counter(hypos).values()) 
        w = entropy(Counter(unique_words).values())
        p.add(len(hypos), [('diversity', d),('word_entropy', w),('avg_dist_hypo', avg_dist_hypo), ('avg_dist_prem', avg_dist_prem)])
    arrd = p.sum_values['diversity']
    arrw = p.sum_values['word_entropy']
    arrj = p.sum_values['avg_dist_hypo']
    arrp = p.sum_values['avg_dist_prem']
    
    return arrd[0] / arrd[1], arrw[0] / arrw[1], arrj[0] / arrj[1],  arrp[0] / arrp[1]

def cut_zeros(list):
    return [a for a in list if a > 0]

class ValidateGen(Callback):
    
    def __init__(self, dev, gen_test, beam_size, hypo_len, samples, noise_size, 
                 glove, cmodel, adverse, diverse):
        self.dev  = dev        
        self.gen_test=gen_test
        self.beam_size = beam_size
        self.hypo_len = hypo_len
        self.samples = samples
        self.noise_size = noise_size
        self.cmodel= cmodel
        self.glove = glove
        self.adverse = adverse    
        self.diverse = diverse
    def on_epoch_end(self, epoch, logs={}):
        gm.update_gen_weights(self.gen_test[0], self.model)        
        val_log =  validate(self.dev, self.gen_test, self.beam_size, self.hypo_len, self.samples,
                 self.noise_size, self.glove, self.cmodel, self.adverse, self.diverse)
        logs.update(val_log)
        

                            

    
    
    
    

                             
    
    
