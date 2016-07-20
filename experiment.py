import load_data
import generative_models as gm
import generative_alg as ga
import classify_models as cm
import classify_alg as ca
import adverse_alg as aa
import augment

import sys
import csv
import glob
import numpy as np
import os.path

if __name__ == "__main__":
    train, dev, test, wi, glove, prem_len, hypo_len = load_data.main()
    
    method = sys.argv[1]
    version = int(sys.argv[2])
    g_hidden_size = int(sys.argv[3])
    latent_size = int(sys.argv[4])

    c_hidden_size = 150
    a_hidden_size = 150
    beam_size = 1
    batch_size = 64
    gen_epochs = 20
    div_per_premise = 64
    div_samples = 32
    augment_file_size = 2 ** 15
    aug_threshold = 0.9
    thresholds = [0.0, 0.3, 0.6, aug_threshold] #, True, 'la0.5', 'lb0.8', 'aa0.25', 'ab0.25'] 
    epoch_size = (len(train[0]) / batch_size) * batch_size
    dev_sample_size = (len(dev[0]) / batch_size) * batch_size

    dir_name = 'models/real' + str(version) + '-' + str(g_hidden_size) + '-' + str(latent_size)
    
    orig_cmodel_dir = 'models/cmodel/'
    cmodel = cm.attention_model(c_hidden_size, glove)
    if os.path.exists(orig_cmodel_dir):
       cmodel.load_weights(orig_cmodel_dir + 'model.weights')
    
    if method == 'orig_class':
        ca.train(train, dev, cmodel, orig_cmodel_dir, batch_size)
    
    if method == 'train_gen':
        gtrain = gm.gen_train(len(train[0]), g_hidden_size, latent_size, glove, hypo_len, version)
        ga.train(train, dev, gtrain, dir_name, batch_size, glove, beam_size, 
               epoch_size, dev_sample_size, cmodel, gen_epochs)

    if method == 'augment':
        gtrain = gm.gen_train(len(train[0]), g_hidden_size, latent_size, glove, hypo_len, version)
        gtrain.load_weights(dir_name + '/weights.hdf5')
        gtest = gm.gen_test(gtrain, glove, batch_size)
        augment.new_generate_save(dev, dir_name, augment_file_size, gtest, beam_size, hypo_len, 
                                  latent_size, cmodel, wi, 'dev', len(dev[0]), aug_threshold)
        augment.new_generate_save(train, dir_name, augment_file_size, gtest, beam_size, hypo_len,
                                  latent_size, cmodel, wi, 'train', len(train[0]), aug_threshold)
     
    if method == 'train_class':
       for t in thresholds:
           if type(t) == str and t[0] == 'a':
               aug_train, aug_dev = augment.load_dataset(dir_name, True, 2**30, 2**30, wi, prem_len, hypo_len)
               aug_dev = aa.filter_adverserial(aug_dev, t, len(dev[0]), dir_name, glove, a_hidden_size)
               aug_train = aa.filter_adverserial(aug_train, t, len(train[0]), dir_name, glove, a_hidden_size)
           else:
               aug_train, aug_dev = augment.load_dataset(dir_name, t, len(train[0]), len(dev[0]), wi, prem_len, hypo_len)
           
           aug_cmodel = cm.attention_model(c_hidden_size, glove)
           ca.train(aug_train, aug_dev, aug_cmodel, dir_name + '/threshold' + str(t), batch_size)

    if method == 'train_discriminator':
        aug_train, aug_dev = augment.load_dataset(dir_name, 0.0, len(train[0]), len(dev[0]), wi, prem_len, hypo_len)
        aa.adverse_model_train(dir_name, train, aug_train, dev, aug_dev, a_hidden_size, glove)
      
    if method == 'evaluate':
        csvf =  open(dir_name + '/total_eval.csv', 'wb')
        writer = csv.writer(csvf)
        writer.writerow(['threshold', 'total_params', 'atrain_params', 'class_loss', 'class_entropy', 'class_acc', 'neutr_acc', 
                         'contr_acc','ent_acc', 'adverse_acc', 'sent_div', 'word_div', 'hypo_dist', 'prem_dist', 
                          'loss_dev', 'acc_dev', 'loss_test', 'acc_test', 'aug_dev_acc', 'avg_loss'])
       
        gtrain = gm.gen_train(len(train[0]), g_hidden_size, latent_size, glove, hypo_len, version)
        gtrain.load_weights(dir_name + '/weights.hdf5')
        gtest = gm.gen_test(gtrain, glove, batch_size)
        params = gm.count_params(gtrain)
        tot_par = params['total'] - params['non_trainable']
        atrain_par = params['trainable']
        sent_div, word_div, hypo_dist, prem_dist = ga.diversity(dev, gtest, beam_size, hypo_len,
                                                        latent_size, div_per_premise, div_samples)
 
        rec_thresholds = load_data.load_thresholds(dir_name)
        aug_cmodel = cm.attention_model(c_hidden_size, glove)
        for t in rec_thresholds:
            print t
            aug_train, aug_dev = augment.load_dataset(dir_name, t, len(train[0]), len(dev[0]), wi, prem_len, hypo_len)
            
            preds = cmodel.predict(list(aug_dev[:2]))
            centropy = np.mean(-np.sum(preds * np.log(preds), axis=1))
            cacc = np.mean(np.argmax(preds, axis = 1) == np.argmax(aug_dev[2], axis =1))
            closs = -np.mean(np.sum(aug_dev[2] * np.log(preds), axis = 1))
            neutr_acc = np.mean(np.argmax(preds[aug_dev[2][:,0] == 1], axis = 1) == 0) 
            contr_acc = np.mean(np.argmax(preds[aug_dev[2][:,1] == 1], axis = 1) == 1)
            ent_acc = np.mean(np.argmax(preds[aug_dev[2][:,2] == 1], axis = 1) == 2)
            adverse_acc = aa.adverse_model_validate(dir_name, dev, aug_dev, glove, a_hidden_size)
            
            filename = dir_name + '/threshold' + str(t) + '/model.weights'
            if os.path.isfile(filename):
                aug_cmodel.load_weights(filename)
                loss_dev, acc_dev, = aug_cmodel.evaluate([dev[0], dev[1]], dev[2])
                loss_test, acc_test = aug_cmodel.evaluate([test[0], test[1]], test[2])
                aug_dev_loss, aug_dev_acc = aug_cmodel.evaluate([aug_dev[0], aug_dev[1]], aug_dev[2])
            avg_loss = np.mean(aug_dev[3])
            row = [t, tot_par, atrain_par, closs, centropy, cacc, neutr_acc, contr_acc, ent_acc, adverse_acc,
                   sent_div, word_div, hypo_dist, prem_dist,
                   loss_dev, acc_dev, loss_test, acc_test, aug_dev_acc, avg_loss]
            str_row = [str(t)] + ["%0.4f" % stat for stat in row[1:]]
            writer.writerow(str_row)           
    if method == 'load_gen':
        gtrain = gm.gen_train(len(train[0]), g_hidden_size, latent_size, glove, hypo_len, version)
        gtrain.load_weights(dir_name + '/weights.hdf5')
        gtest = gm.gen_test(gtrain, glove, batch_size)

            
