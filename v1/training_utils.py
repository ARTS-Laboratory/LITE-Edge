import numpy as np
import tensorflow.keras as keras

'''
Generate data batches for training and validation.
'''
class TrainingGenerator(keras.utils.Sequence):
    
    def __init__(self, *args, train_len=400):
        self.args = args
        self.train_len = train_len
        self.length = args[0].shape[1]//train_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        rtrn = [arg[:,index*self.train_len:(index+1)*self.train_len,:] for arg in self.args]
        return rtrn[:-1], rtrn[-1]

# Returns a version of the package data aligned to the reference data
def sync_data(reference, package):
    correlation = np.correlate(reference, package, mode='full')
    lags = np.arange(-len(reference) + 1, len(reference))
    
    return np.roll(package, -1 * lags[np.argmax(correlation)])
