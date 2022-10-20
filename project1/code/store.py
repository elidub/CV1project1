import os
import pickle

from stl10_input import DATA_DIR

### Save and load utils

def storage_path(name, percentage1, percantage2):
    file_path = os.path.join(DATA_DIR, f'storage/{name}_{percentage1}_{percantage2}.pkl')
    return file_path

def load(name, percentage1, percantage2):
    file_path = storage_path(name, percentage1, percantage2)
    file = pickle.load(open(file_path, 'rb')) #To load saved model from local directory
    return file

def save(file, name, percentage1, percantage2):
    file_path = storage_path(name, percentage1, percantage2)
    pickle.dump(file, open(file_path, 'wb')) #Saving the model
    
    


