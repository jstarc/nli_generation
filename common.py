import numpy as np
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
import csv

def make_fixed_embeddings(glove, seq_len):
    glove_mat = np.array(glove.values())
    return Embedding(input_dim = glove_mat.shape[0], output_dim = glove_mat.shape[1], 
                       weights = [glove_mat], trainable = False, input_length  = seq_len)
                       
   
class CsvHistory(Callback):
    
    def __init__(self, filename):
        self.file = open(filename, 'a', 0)
        self.writer = csv.writer(self.file)
        self.header = True
 
    def on_epoch_end(self, epoch, logs={}):
        if self.header:
            self.writer.writerow(['epoch'] + logs.keys())
            self.header = False
        self.writer.writerow([epoch] + ["%0.4f" % v for v in logs.values()])

    def on_train_end(self, logs={}):
        self.file.close()  

def merge_result_batches(batches):
    res = list(batches[0])
    for i in range(1, len(batches)):
        for j in range(len(res)):
            res[j] = np.concatenate([res[j], batches[i][j]])
    return res
    

