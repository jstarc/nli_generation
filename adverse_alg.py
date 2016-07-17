import numpy as np
import os 

import load_data
from keras.callbacks import ModelCheckpoint, EarlyStopping
import adverse_models as am        

def train_adverse_model(train, dev, adverse_model, generative_model, word_index, model_dir, 
                        nb_epochs, batch_size, hypo_len): 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    train_gen = adverse_generator(train, generative_model, word_index, 0.8, batch_size, 
                                  hypo_len)
    dev_gen = adverse_generator(dev, generative_model, word_index, 0.0, batch_size, hypo_len)
    val_data = prepare_dev_data(dev_gen, len(dev) / batch_size)
    saver = ModelCheckpoint(model_dir + '/adverse.weights', monitor = 'loss')
    es = EarlyStopping(patience = 5)
    
    return adverse_model.fit_generator(train_gen, samples_per_epoch = 64000, 
                 nb_epoch = nb_epochs, callbacks = [saver, es], validation_data = val_data) 
   
def prepare_dev_data(dev_gen, batches):
    dev_data = [next(dev_gen) for _ in range(batches)]
    trains = np.vstack([data[0][0] for data in dev_data])
    gens = np.vstack([data[0][1] for data in dev_data])
    ys = np.concatenate([data[1] for data in dev_data])
    return [trains, gens], ys
    
    
def adverse_generator(train, gen_model, word_index, cache_prob, batch_size, hypo_len):
    cache =  []    
    while True:
         mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        
         for i, train_index in mb:
             if len(train_index) != batch_size:
                 continue
             
             orig_batch = [train[k] for k in train_index]
             if np.random.random() > cache_prob or len(cache) < 100:
                 gen_batch, _ = make_gen_batch(orig_batch, gen_model, word_index, hypo_len)
                 cache.append(gen_batch)
             else:
                 gen_batch = cache[np.random.random_integers(0, len(cache) - 1)]
                 
             train_batch = make_train_batch(orig_batch, word_index, hypo_len)
             yield [train_batch, gen_batch], np.zeros(batch_size)
        
def make_train_batch(orig_batch, word_index, hypo_len):
    _, X_hypo, _ = load_data.prepare_split_vec_dataset(orig_batch, word_index.index)
    return load_data.pad_sequences(X_hypo, maxlen = hypo_len, dim = -1, padding = 'post')
    

def manual_tester(dev, aug_dev, discriminator, word_index, target_size, filename):
    
    true_ex = 0
    with open(filename+'.hidden', 'w') as h, open(filename+'.revealed', 'w') as r:
        indices = np.random.random_integers(0, len(dev[0]) - 1, target_size)
        for i in indices:
            gen_ex = np.array([aug_dev[1][i]])
            org_ex = np.array([dev[1][i]])
            gen_pred = discriminator.predict(gen_ex)[0][0]
            train_pred = discriminator.predict(org_ex)[0][0]
            
            gen_hypo = word_index.print_seq(aug_dev[1][i])
            train_hypo = word_index.print_seq(dev[1][i])
                
            r_data = [gen_hypo, train_hypo, str(gen_pred), str(train_pred)]
            r.write("\t".join(r_data) + '\n')

            h_data = [gen_hypo, train_hypo] if np.random.rand() > 0.5 else [train_hypo, gen_hypo]
            h.write("\t".join(h_data) + '\n')
            
            if train_pred > gen_pred:
                true_ex += 1

        print true_ex / float(target_size)               
def adverse_model_train(model_dir, train, aug_train, dev, aug_dev, dim, glove):
    discriminator = am.discriminator(glove, dim)
    ad_model = am.adverse_model(discriminator)
    dev_len = len(aug_dev[1])
    res = ad_model.fit([train[1], aug_train[1]], np.zeros(len(train[1])),
                       validation_data=([dev[1][:dev_len], aug_dev[1]], np.zeros(dev_len)),
                       verbose = 1, nb_epoch = 5)
    discriminator.save_weights(model_dir + '/adverse.weights')
    
def adverse_model_validate(model_dir, dev, aug_dev, glove, dim):
    discriminator = am.discriminator(glove, dim)
    discriminator.load_weights(model_dir + '/adverse.weights')
    dev_len = len(aug_dev[1])
    preds_orig = discriminator.predict(dev[1][:dev_len])
    preds_aug =  discriminator.predict(aug_dev[1])

    return np.mean(preds_orig > preds_aug)

def init_adverserial_threshold_function(target_dir, ad_threshold, glove, dim, wi, hypo_len):
    ### too slow to use
    discriminator = am.discriminator(glove, dim)
    discriminator.load_weights(target_dir + '/adverse.weights')
    padding_example = (['a'] * 25, ['a'] * 15, 'entailment')
    def t_func(example, loss, cpred, ctrue):
        if not ctrue:
            return False
        vec_ex = load_data.prepare_split_vec_dataset([example, padding_example], wi.index, True)
        pred = discriminator.predict(vec_ex[1][:1])[0][0]
        return pred > ad_threshold
    return t_func


def filter_adverserial(dataset, threshold, target_size, target_dir, glove, dim):
    arg = threshold[1] == 'b' 
    num = float(threshold[2:])
    label_size = target_size / 3
    discriminator = am.discriminator(glove, dim)
    discriminator.load_weights(target_dir + '/adverse.weights')
    ad_preds = discriminator.predict(dataset[1], verbose = 1)[:,0]
    indices =  (ad_preds < num) if arg else (ad_preds > num)
    final_indices = []
    for l in range(3):
        label_args = dataset[2][:, l] == 1
        final_indices += list(np.where(indices * label_args)[0][:label_size])
    ind = np.sort(final_indices)
    return (dataset[0][ind], dataset[1][ind], dataset[2][ind])
        
        
        
