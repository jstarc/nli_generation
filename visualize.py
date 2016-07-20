import numpy as np
import load_data
from generative_alg import *
from keras.utils.generic_utils import Progbar
from load_data import load_word_indices
from keras.preprocessing.sequence import pad_sequences

import pandas as pa
import augment

def test_points(premises, labels, noises, gtest, cmodel, hypo_len):
    p = Progbar(len(premises))
    hypos = []
    bs = 64 
    for i in range(len(labels) / bs):
        words, _  = generative_predict_beam(gtest, premises[i * bs: (i+1)*bs], 
                          noises[i * bs: (i+1)*bs,None,:], labels[i * bs: (i+1)*bs], True, hypo_len)
        hypos.append(words)
        p.add(len(words))
    hypos = np.vstack(hypos)
    cpreds = cmodel.evaluate([premises[:len(hypos)], hypos], labels[:len(hypos)])
    print cpreds


def print_hypos(premise, label, gen_test, beam_size, hypo_len, noise_size, wi):
    words = single_generate(premise, label, gen_test, beam_size, hypo_len, noise_size)
    batch_size = gen_test[0].input_layers[0].input_shape[0]

    per_batch  = batch_size / beam_size
    premises = [premise] * per_batch
    noise_input = np.random.normal(scale=0.11, size=(per_batch, 1, noise_size))
    class_indices = np.ones(per_batch) * label
    class_indices = load_data.convert_to_one_hot(class_indices, 3)
    words, loss = generative_predict_beam(gen_test, premises, noise_input,
                             class_indices, True, hypo_len)
    
    print 'Premise:', wi.print_seq(premise)
    print 'Label:', load_data.LABEL_LIST[label]
    print 
    print 'Hypotheses:'
    for h in words:
        print wi.print_seq(h)

def load_sentence(string, wi, len = 25):
    tokens = string.split()
    tokens = load_word_indices(tokens, wi.index)
    return pad_sequences([tokens], maxlen = len, padding = 'pre')[0]
    

def find_true_examples():
    models = ['8-150-2', '8-150-4', '8-150-8', '8-150-16', '8-150-32', '8-150-147', '6-150-8', '7-150-8' ,'9-226-8']
    final_premises = set()
    subset = {}
    for model in models:
         data = pa.read_csv('models/real' + model + '/dev1')
         data = data[data['ctrue']]
         neutr = data[data['label'] == 'neutral']
         contr = data[data['label'] == 'contradiction']
         entail = data[data['label'] == 'entailment']
         subset[model] = [neutr, contr, entail]
         premises = set(neutr['premise']) &  set(contr['premise']) &  set(entail['premise'])
         if len(final_premises) == 0:
             final_premises = premises
         else:
             final_premises &= premises
    final_premises = list(final_premises)

    with open('results/ext_examples.txt', 'w') as fi:
        for i in range(len(final_premises)):
            premise = final_premises[i]
            fi.write(premise + '\n')
            for m in models:
                fi.write(m + '\n')
                for l in range(3):
                    filtered = subset[m][l][subset[m][l]['premise'] == premise]
                    for f in range(len(filtered)):
                        hypo = filtered['hypo'].iloc[f]
                        label = filtered['label'].iloc[f][:4]
                        fi.write(label + '\t'  + hypo + '\n')
        fi.write('\n')
    
    
  
