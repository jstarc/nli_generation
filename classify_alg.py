import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from common import CsvHistory

def train(train, dev, model, model_dir, batch_size):
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)  
    es = EarlyStopping(patience = 2)
    saver = ModelCheckpoint(model_dir + '/model.weights', monitor = 'val_loss', save_best_only = True)
    csv = CsvHistory(model_dir + '/history.csv')
    return model.fit([train[0], train[1]], train[2], batch_size=batch_size, 
                     nb_epoch = 1000, validation_data = ([dev[0], dev[1]], dev[2]), 
                     callbacks = [saver, es, csv])
        
